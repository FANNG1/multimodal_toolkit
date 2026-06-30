# multimodal_toolkit

Audio call-centre analysis POC: ingest recordings from S3, store audio as Lance blob v2, transcribe with SenseVoice, analyse with DeepSeek, append acoustic embeddings, and query by scalar filter or nearest-neighbour.

## Architecture overview

Two pipeline families live in this repo:

| Package | Entry points | Status |
|---------|--------------|--------|
| `pipeline/` | `mmt-ingest`, `mmt-analyze`, `mmt-embed`, `mmt-query` | Original POC; blobs ingested first, analysis appended in place |
| `workflow/` | `python -m multimodal_toolkit.workflow.<step>` | Revised design; analysis runs first, blobs and metadata co-ingested in one write |

The sections below describe the **workflow** design. See `pipeline/` source for the original approach.

## Workflow data flow

```
Manifest (parquet / jsonl / csv)
  doc_id, s3_url
       │
       ▼  Stage 1 — workflow/analyze.py
       │  Daft: read manifest → download audio bytes → duration filter
       │        → SenseVoice ASR (transcript + acoustic_emotion)
       │        → PII redaction (ID card, phone numbers)
       │        → DeepSeek LLM (downgrade_related, bad_tone, emotion_score …)
       │        → [optional --embed] audio_embedding (128-dim)
       │
       ├── (no --embed)  →  JSONL on S3
       └── (--embed)     →  Lance staging table on S3
                │
                ▼  Stage 2 — workflow/ingest.py
                │  Daft: read JSONL or Lance staging table
                │        → download audio blobs from s3_url
                │        → stamp ingest_time
                │        → write_lance (blob v2, append)
                │        → validate blob v2
                │
                ▼  Lance asset table  (blob v2, local or S3)
                │  columns: doc_id, s3_url, audio_blob,
                │           transcript, acoustic_emotion,
                │           downgrade_related, primary_reason,
                │           secondary_reason, summary, confidence,
                │           text_emotion, bad_tone, emotion_score,
                │           [audio_embedding], ingest_time
                │
                ▼  Stage 3 — workflow/index.py
                │  lance_ray  : IVF_PQ index on audio_embedding
                │  daft_lance : ZONEMAP index on ingest_time
                │
                ├──▶  Stage 4 — workflow/query.py
                │     Daft scanner pushdown  : --where  (scalar filter)
                │     pylance scanner nearest : --vector-from  (ANN)
                │
                └──▶  Stage 5 — workflow/manage.py
                      pylance ds.delete()        : --before / --after
                      lance_ray.compact_files    : automatic after delete
```

### Engine assignments

| Engine         | Used for                                                                              | Reason                                               |
|----------------|---------------------------------------------------------------------------------------|------------------------------------------------------|
| **Daft**       | manifest read, S3 download, ASR/LLM pipeline, Lance write (Stage 1 & 2), scalar query | Primary engine; stable APIs                          |
| **lance_ray**  | IVF_PQ and ZONEMAP index creation, `compact_files`                                    | Preferred for all Lance table management; mature distributed APIs |
| **daft_lance** | `compact_files` fallback; scalar index if lance_ray unavailable                       | Daft-first applies to data processing, not table management |
| **pylance**    | ANN scanner (`nearest=`), row delete, `cleanup_old_versions`                          | Only option for delete and cleanup; exposes `_distance` for ANN |

## Setup

```sh
uv sync --upgrade
```

Create a `.env` file (or export variables directly):

```sh
# S3 / MinIO
MINIO_ENDPOINT=http://127.0.0.1:9000
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin
MINIO_REGION=us-east-1

# LLM — leave blank to skip DeepSeek analysis (fields will be null)
DEEPSEEK_API_KEY=sk-...
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat

# ASR device
ASR_DEVICE=cpu          # or cuda

# Duration filter applied in Stage 1
MIN_DURATION_S=0
MAX_DURATION_S=1800

# Embedding backend used in Stage 1 --embed and pipeline/embed
EMBED_BACKEND=signal    # signal (128-dim RMS+ZCR) or wav2vec2

# Daft runner
USE_RAY=0               # set to 1 to use Ray for Daft-backed steps
RAY_ADDRESS=            # leave empty to start/join local Ray
```

## Usage — workflow pipeline

The manifest must be parquet, jsonl, or csv with at minimum `doc_id` and `s3_url` columns.  
`--lance-uri` accepts both local paths and `s3://` URIs.

### Stage 1 — analyze

Downloads audio from S3, runs ASR and LLM analysis, writes output to S3.

```sh
# Output: JSONL (no embeddings)
python -m multimodal_toolkit.workflow.analyze \
  --manifest s3://bucket/audio/manifest.parquet \
  --out      s3://bucket/audio/analysis.jsonl

# Output: Lance staging table (includes audio_embedding; required for ANN search later)
python -m multimodal_toolkit.workflow.analyze \
  --manifest s3://bucket/audio/manifest.parquet \
  --out      s3://bucket/audio/staging.lance \
  --embed
```

### Stage 2 — ingest

Reads Stage 1 output, downloads audio blobs, and appends them together with the analysis metadata into the Lance asset table.

```sh
python -m multimodal_toolkit.workflow.ingest \
  --analysis  s3://bucket/audio/analysis.jsonl \
  --lance-uri s3://bucket/audio/calls.lance
```

Pass a `.lance` URI to `--analysis` if Stage 1 was run with `--embed`.

### Stage 3 — index

Builds indexes for fast query. Run after the table has enough rows (IVF_PQ needs at least `num_partitions × 256` rows; use `--num-partitions 1` for small tables).

```sh
# Build both indexes (default)
python -m multimodal_toolkit.workflow.index \
  --lance-uri s3://bucket/audio/calls.lance

# Vector index only
python -m multimodal_toolkit.workflow.index \
  --lance-uri s3://bucket/audio/calls.lance \
  --no-time

# Tune partitions for small tables
python -m multimodal_toolkit.workflow.index \
  --lance-uri s3://bucket/audio/calls.lance \
  --num-partitions 1 --num-sub-vectors 8
```

### Stage 4 — query

```sh
# Scalar filter (Daft pushdown)
python -m multimodal_toolkit.workflow.query \
  --lance-uri s3://bucket/audio/calls.lance \
  --where "bad_tone = true OR downgrade_related = true" \
  --top-k 20

# ANN vector search (pylance; exposes _distance)
python -m multimodal_toolkit.workflow.query \
  --lance-uri s3://bucket/audio/calls.lance \
  --vector-from call_001.mp3 \
  --top-k 10

# Combined: ANN with scalar pre-filter
python -m multimodal_toolkit.workflow.query \
  --lance-uri s3://bucket/audio/calls.lance \
  --vector-from call_001.mp3 \
  --where "downgrade_related = true" \
  --top-k 10
```

### Stage 5 — manage

Delete rows by ingest date and compact the table:

```sh
# Delete rows ingested before a date
python -m multimodal_toolkit.workflow.manage \
  --lance-uri s3://bucket/audio/calls.lance \
  --before 2025-01-01

# Delete rows outside a date window
python -m multimodal_toolkit.workflow.manage \
  --lance-uri s3://bucket/audio/calls.lance \
  --after 2024-06-01 --before 2024-12-31
```

Compaction and version cleanup run automatically after delete.

## Usage — original pipeline (mmt-*)

The `pipeline/` entry points use a different ordering: blobs are ingested first, then analysis and embeddings are appended as separate steps.

```sh
mmt-ingest  --manifest s3://bucket/audio/manifest.parquet \
            --lance-uri /tmp/calls.lance

mmt-analyze --lance-uri /tmp/calls.lance \
            --out-jsonl /tmp/analysis.json

mmt-embed   --lance-uri /tmp/calls.lance

mmt-query   --lance-uri /tmp/calls.lance \
            --where "bad_tone = true" \
            --top-k 5

mmt-query   --lance-uri /tmp/calls.lance \
            --query-doc-id call_001.mp3 \
            --top-k 5
```

## Verified versions

| Component | Version | Notes |
|-----------|---------|-------|
| Daft | 0.7.15 | Main execution engine |
| daft-lance | 0.4.0 | `read_lance`, `write_lance`, `take_blobs`, `create_scalar_index`, `compact_files` |
| pylance | 7.0.0 | Lance dataset, blob v2, ANN scanner, delete, cleanup |
| lance-ray | 0.4.2 | Vector index creation; write-back path deferred |
| Ray | 2.55.1 | Pulled in by lance-ray; Daft uses native runner unless `USE_RAY=1` |

Default Daft runner: `native` (local multi-threaded). Set `USE_RAY=1` to switch to Ray for Daft-backed steps. Stage 3 (lance_ray index) and Stage 4 ANN (pylance scanner) always run locally regardless of `USE_RAY`.

## Notes and known limitations

**Audio downloaded twice in the workflow pipeline.**  
Stage 1 downloads audio bytes for ASR and embedding; Stage 2 downloads the same files again to store as Lance blobs. This is intentional — analysis output (JSONL) does not carry raw bytes across stages. Plan accordingly for bandwidth costs or cache files locally between stages.

**`--embed` in Stage 1 is required for ANN search.**  
If Stage 1 was run without `--embed`, the Lance asset table has no `audio_embedding` column. Stage 3 will error and Stage 4 `--vector-from` will have nothing to search. Re-run Stage 1 with `--embed` and re-ingest to add embeddings.

**IVF_PQ minimum row count.**  
The default `--num-partitions 16` requires at least 4096 rows. For tables with fewer rows, pass `--num-partitions 1` (or skip `--embedding` and rely on scalar queries only).

**DeepSeek key absent → LLM columns are null.**  
If `DEEPSEEK_API_KEY` is not set, `downgrade_related`, `bad_tone`, `primary_reason`, `summary`, `confidence`, `text_emotion`, and `emotion_score` are all `null`. ASR and acoustic embeddings still run normally.

**Blob v2 is validated after every ingest.**  
`validate_blob_v2` raises immediately if Lance silently downgraded `audio_blob` to `large_binary`. Never skip this check when testing new library versions.

**Lance write-back for embeddings via daft_lance / lance-ray is deferred.**  
`daft_lance.merge_columns_df` has correctness issues with blob v2 columns in this POC. The current `pipeline/embed.py` uses pylance `add_columns` instead. The distributed lance-ray write-back path is deferred until a newer stable release.

**Local Lance URIs are verified end-to-end.**  
S3 Lance table write/read is exercised by the underlying libraries but should be treated as a separate validation item for this POC.
