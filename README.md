# multimodal_toolkit

Audio call-centre analysis POC: ingest recordings from S3, store audio as Lance blob v2, transcribe with SenseVoice, analyse with DeepSeek, append acoustic embeddings, and query by scalar filter or nearest-neighbour.

Also includes an image analysis workflow (face presence + clarity detection), fully isolated from the audio pipeline — see [Image workflow](#image-workflow) below.

## Architecture overview

Media-specific analysis lives under each media package (`audio/workflow/` and
`image/workflow/`). Top-level `workflow/` owns media-agnostic Lance table
operations such as index creation and retention management. Shared S3 / Ray
configuration stays in `multimodal_toolkit/config.py`; media-specific settings
and schemas live in `audio/config.py`, `image/config.py`, `audio/schema.py`,
and `image/schema.py`.

The audio pipeline runs analysis first, then audio blobs and analysis metadata
are co-ingested into the Lance asset table.

## Workflow data flow

```
Manifest (parquet / jsonl / csv)
  doc_id, s3_url
       │
       ▼  Stage 1 — audio/workflow/analyze.py
       │  Daft: read manifest → download audio bytes → duration filter
       │        → SenseVoice ASR (transcript + acoustic_emotion)
       │        → PII redaction (ID card, phone numbers)
       │        → DeepSeek LLM (downgrade_related, bad_tone, emotion_score …)
       │        → [optional --embed] audio_embedding (128-dim)
       │
       ├── (no --embed)  →  JSONL on S3
       └── (--embed)     →  Lance staging table on S3
                │
                ▼  Stage 2 — audio/workflow/ingest.py
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
                │  pylance    : ZONEMAP index on ingest_time
                │
                ├──▶  Stage 4 — audio/workflow/query.py
                │     Daft SQL (daft.sql())   : --sql    (scalar / aggregation)
                │     Daft scanner pushdown   : --where  (scalar filter)
                │     Daft scanner nearest    : --vector-from  (ANN, IVF index)
                │
                └──▶  Stage 5 — workflow/manage.py
                      pylance ds.delete()        : --before / --after
                      lance_ray.compact_files    : automatic after delete
```

### Engine assignments

| Engine         | Used for                                                                              | Reason                                                            |
|----------------|---------------------------------------------------------------------------------------|-------------------------------------------------------------------|
| **Daft**       | manifest read, S3 download, ASR/LLM pipeline, Lance write (Stage 1 & 2), scalar and ANN query | Primary engine; stable APIs                                |
| **lance_ray**  | IVF_PQ vector index creation, `compact_files`                                         | Preferred for Lance table management; distributed Ray workers     |
| **pylance**    | ZONEMAP scalar index, row delete, `cleanup_old_versions`                              | ZONEMAP: lance_ray requires unreleased code; delete: only API     |
| **daft_lance** | fallback for `compact_files` if lance_ray unavailable                                 | Not used for index; Daft-first applies to data processing only    |

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

# Embedding backend used in Stage 1 --embed
EMBED_BACKEND=signal    # signal (128-dim RMS+ZCR) or wav2vec2

# Daft runner
USE_RAY=0               # set to 1 to use Ray for Daft-backed steps
RAY_ADDRESS=            # leave empty to start/join local Ray
```

## Usage — audio workflow pipeline

The manifest must be parquet, jsonl, or csv with at minimum `doc_id` and `s3_url` columns.  
`--lance-uri` accepts both local paths and `s3://` URIs.

### Stage 1 — analyze

Downloads audio from S3, runs ASR and LLM analysis, writes output to S3.

```sh
# Output: JSONL (no embeddings)
python -m multimodal_toolkit.audio.workflow.analyze \
  --manifest s3://bucket/audio/manifest.parquet \
  --out      s3://bucket/audio/analysis.jsonl

# Output: Lance staging table (includes audio_embedding; required for ANN search later)
python -m multimodal_toolkit.audio.workflow.analyze \
  --manifest s3://bucket/audio/manifest.parquet \
  --out      s3://bucket/audio/staging.lance \
  --embed
```

### Stage 2 — ingest

Reads Stage 1 output, downloads audio blobs, and appends them together with the analysis metadata into the Lance asset table.

```sh
python -m multimodal_toolkit.audio.workflow.ingest \
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
# Scalar filter (Daft pushdown via read_lance default_scan_options)
python -m multimodal_toolkit.audio.workflow.query \
  --lance-uri s3://bucket/audio/calls.lance \
  --where "bad_tone = true OR downgrade_related = true" \
  --top-k 20

# Full Daft SQL SELECT (table name in scope: calls)
python -m multimodal_toolkit.audio.workflow.query \
  --lance-uri s3://bucket/audio/calls.lance \
  --sql "SELECT primary_reason, COUNT(*) AS cnt FROM calls GROUP BY primary_reason ORDER BY cnt DESC"

# Scalar filter with projection via Daft SQL
python -m multimodal_toolkit.audio.workflow.query \
  --lance-uri s3://bucket/audio/calls.lance \
  --sql "SELECT doc_id, emotion_score, primary_reason FROM calls WHERE bad_tone = true AND emotion_score > 0.5 ORDER BY emotion_score DESC" \
  --top-k 20

# ANN vector search via Daft Lance scanner (uses IVF index)
python -m multimodal_toolkit.audio.workflow.query \
  --lance-uri s3://bucket/audio/calls.lance \
  --vector-from call_001.mp3 \
  --top-k 10

# Combined: ANN with scalar pre-filter
python -m multimodal_toolkit.audio.workflow.query \
  --lance-uri s3://bucket/audio/calls.lance \
  --vector-from call_001.mp3 \
  --where "downgrade_related = true" \
  --distance-min 0.0 \
  --distance-max 1.0 \
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

## Image workflow

Image analysis lives in `multimodal_toolkit/image/` and is fully isolated from the audio
pipeline (own workflow entry points, own Lance asset table) so the two can evolve independently.
Stage 3 (index) and Stage 5 (manage) are media-agnostic and shared.

v1 detections, all local (no VLM/API):

| Detection | Method | Output columns |
|-----------|--------|----------------|
| Face presence | InsightFace SCRFD (`buffalo_l`, detection module only, CPU) | `face_count`, `face_score`, `face_area_ratio`, `has_face` |
| Clarity / blur | OpenCV Laplacian variance on the image resized to `IMAGE_LONG_EDGE` (whole image + largest-face crop) | `blur_score`, `face_blur_score`, `is_blurry`, `is_face_blurry` |
| Text/image similarity | ChineseCLIP (`OFA-Sys/chinese-clip-vit-base-patch16`) | `image_embedding` |

All face-derived metrics (`face_score`, `face_area_ratio`, `face_blur_score`) come from
the same face — the largest one — so the rule engine's AND conditions always judge a
single face rather than mixing metrics from different detections.

Boolean verdicts are derived from raw scores by a threshold rule engine
(`image/rules.py`); both raw scores and verdicts are persisted, so thresholds can be
re-tuned via SQL without re-running the models. The lower bound for such retuning is
the detector's own coarse filter `FACE_DET_THRESH` (default 0.3) — faces below it never
reach the table.

Every manifest entry produces exactly one output row. Failed items are kept with a
`status` column (`ok` / `download_failed` / `decode_failed`), null scores, and null
verdicts — "unknown" stays distinguishable from "judged no", and unreadable images are
themselves a reportable compliance signal.

Environment variables (all optional):

```sh
INSIGHTFACE_MODEL=buffalo_l   # insightface model pack
INSIGHTFACE_ROOT=             # pre-baked model dir for offline/container use ("" = ~/.insightface)
FACE_DET_SIZE=640             # SCRFD detection input size
FACE_DET_THRESH=0.3           # SCRFD coarse filter; keep well below FACE_DET_SCORE_MIN
IMAGE_LONG_EDGE=1024          # resize long edge before detection/blur (never upscales)
FACE_DET_SCORE_MIN=0.5        # min det score (of the largest face) for has_face
MIN_FACE_RATIO=0.01           # min face-area / image-area for has_face
BLUR_THRESHOLD=100.0          # blur_score below this → is_blurry
FACE_BLUR_THRESHOLD=80.0      # face_blur_score below this → is_face_blurry
IMAGE_EMBED_MODEL=OFA-Sys/chinese-clip-vit-base-patch16
IMAGE_EMBED_DEVICE=cpu
IMAGE_EMBED_DIM=512
```

The first detection run downloads the SCRFD model pack (~280 MB) to `~/.insightface`; set
`INSIGHTFACE_ROOT` to a pre-populated directory to skip the download. The first
`--embed` run downloads the ChineseCLIP model via Transformers cache.

```sh
# Seed images and write the manifest
python scripts/init_s3.py --media image --data-dir data/images \
  --raw-prefix raw/images --manifest-key image_poc/manifest.parquet

# Stage 1 — analyze (face presence + clarity scores + rule verdicts → JSONL)
python -m multimodal_toolkit.image.workflow.analyze \
  --manifest s3://contacts/image_poc/manifest.parquet \
  --out      s3://contacts/image_poc/analysis.jsonl

# Stage 1 with embeddings — required for text-to-image and image similarity search
python -m multimodal_toolkit.image.workflow.analyze \
  --manifest s3://contacts/image_poc/manifest.parquet \
  --out      s3://contacts/image_poc/staging.lance \
  --embed

# Stage 2 — ingest (image blobs + analysis metadata → Lance image asset table)
python -m multimodal_toolkit.image.workflow.ingest \
  --analysis  s3://contacts/image_poc/staging.lance \
  --lance-uri s3://contacts/image_poc/assets.lance

# If Stage 1 was run without --embed, pass analysis.jsonl instead.

# Stage 3 — index for text/image similarity search
python -m multimodal_toolkit.workflow.index \
  --lance-uri s3://contacts/image_poc/assets.lance \
  --embedding-column image_embedding

# Stage 4 — query (table name in SQL scope: images)
python -m multimodal_toolkit.image.workflow.query \
  --lance-uri s3://contacts/image_poc/assets.lance \
  --where "has_face = true AND is_blurry = false"

python -m multimodal_toolkit.image.workflow.query \
  --lance-uri s3://contacts/image_poc/assets.lance \
  --sql "SELECT doc_id, blur_score, face_count FROM images ORDER BY blur_score ASC"

# Text-to-image search
python -m multimodal_toolkit.image.workflow.query \
  --lance-uri s3://contacts/image_poc/assets.lance \
  --text "头像" \
  --where "status = 'ok'"

# Image similarity search from a local file or an existing table row
python -m multimodal_toolkit.image.workflow.query \
  --lance-uri s3://contacts/image_poc/assets.lance \
  --image-path ./query.jpg

python -m multimodal_toolkit.image.workflow.query \
  --lance-uri s3://contacts/image_poc/assets.lance \
  --image-from face_001.jpg

# Join a description table (doc_id → description) into similarity results.
# The table can be a plain parquet/jsonl/csv file — no ingestion needed:
#
#   import pyarrow as pa, pyarrow.parquet as pq
#   pq.write_table(pa.table({
#       "doc_id": ["face_001.jpg", "group_photo.jpg"],
#       "description": ["清晰正面人像", "两人合影"],
#   }), "descriptions.parquet")
#
# Results gain a `description` column (left join: images without a
# description stay in the results with description = null).
python -m multimodal_toolkit.image.workflow.query \
  --lance-uri s3://contacts/image_poc/assets.lance \
  --text "合影" \
  --desc-table descriptions.parquet

# With --sql the description table is registered as `descriptions`:
python -m multimodal_toolkit.image.workflow.query \
  --lance-uri s3://contacts/image_poc/assets.lance \
  --sql "SELECT i.doc_id, d.description FROM images i LEFT JOIN descriptions d ON i.doc_id = d.doc_id WHERE i.has_face = true" \
  --desc-table descriptions.parquet

# Stage 5 — manage (shared entry point)
python -m multimodal_toolkit.workflow.manage \
  --lance-uri s3://contacts/image_poc/assets.lance --before 2025-01-01
```

## Verified versions

| Component | Version | Notes |
|-----------|---------|-------|
| Daft | 0.7.15 | Main execution engine |
| daft-lance | 0.4.0 | `read_lance`, `write_lance`, `take_blobs`, `create_scalar_index`, `compact_files` |
| pylance | 8.0.0 | Lance dataset, blob v2, ANN scanner, delete, cleanup, blob-table compaction |
| lance-ray | 0.4.2 | Vector index creation; write-back path deferred |
| Ray | 2.55.1 | Pulled in by lance-ray; Daft uses native runner unless `USE_RAY=1` |

Default Daft runner: `native` (local multi-threaded). Set `USE_RAY=1` to switch to Ray for Daft-backed steps. Stage 3 (lance_ray index) and Stage 4 ANN (pylance scanner) always run locally regardless of `USE_RAY`.

## Notes and known limitations

**Audio downloaded twice in the workflow pipeline.**  
Stage 1 downloads audio bytes for ASR and embedding; Stage 2 downloads the same files again to store as Lance blobs. This is intentional — analysis output (JSONL) does not carry raw bytes across stages. Plan accordingly for bandwidth costs or cache files locally between stages.

**`--embed` in Stage 1 is required for ANN search.**  
If Stage 1 was run without `--embed`, the Lance asset table has no `audio_embedding` column. Stage 3 will error and Stage 4 `--vector-from` will have nothing to search. Re-run Stage 1 with `--embed` and re-ingest to add embeddings.
For image tables, keep all batches for the same Lance table consistent: either all Stage 1 runs use `--embed`, or none do. The image ingest step rejects appending `image_embedding` batches to a table created without that column, and vice versa.

**IVF_PQ minimum row count.**  
The default `--num-partitions 16` requires at least 4096 rows. For tables with fewer rows, pass `--num-partitions 1` (or skip `--embedding` and rely on scalar queries only).

**DeepSeek key absent → LLM columns are null.**  
If `DEEPSEEK_API_KEY` is not set, `downgrade_related`, `bad_tone`, `primary_reason`, `summary`, `confidence`, `text_emotion`, and `emotion_score` are all `null`. ASR and acoustic embeddings still run normally.

**Blob v2 is validated after every ingest.**  
`validate_blob_v2` raises immediately if Lance silently downgraded `audio_blob` to `large_binary`. Never skip this check when testing new library versions.

**Local Lance URIs are verified end-to-end.**  
S3 Lance table write/read is exercised by the underlying libraries but should be treated as a separate validation item for this POC.

**Blob v2 compaction requires pylance ≥ 8.0.0.**  
Older pylance (≤ 7.x) fails inside the decoder when compacting tables with blob v2
columns (lance-format/lance#7071, fixed by #7017 and released in 8.0.0). The project
pins `pylance>=8.0.0`; Stage 5 compaction now hard-fails on error instead of being
skipped. The explicit `compaction_options={}` remains as a workaround for a separate
lance-ray 0.4.x defect (lance-format/lance-ray#5224).
