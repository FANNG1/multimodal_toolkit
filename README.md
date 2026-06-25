# multimodal_toolkit

Audio call-centre analysis POC: ingest recordings from S3, store audio as Lance blob v2, transcribe with SenseVoice, analyse with DeepSeek, append acoustic embeddings, and query by scalar filter or nearest-neighbour.

## Pipeline

| Step | Module | What it does |
|------|--------|--------------|
| 1 ingest | `ingest` | Read manifest → Daft S3 download → write Lance blob v2 table |
| 2 analyze | `analyze` | `take_blobs` → duration filter → SenseVoice ASR → PII redaction → DeepSeek LLM → write JSON output |
| 3 embed | `embed` | Lance native `read_blobs` → acoustic signal embedding (128-dim RMS+ZCR) → Lance native `add_columns` |
| 4 query | `query` | Scalar filter via Daft or ANN via Lance native |

## Engine decisions

| Engine | Used for | Reason |
|--------|----------|--------|
| **Daft** | manifest read, S3 download, Lance write, blob materialization (`daft_lance.take_blobs`), scalar query, ASR/LLM analysis pipeline | Primary compute engine where current APIs are stable |
| **Lance native** | embedding blob reads, `add_columns`, ANN query (`scanner(nearest=...)`) | Stable path for blob v2 bytes, appending new columns, and exposing `_distance` |
| **lance-ray** | future distributed embedding/write-back path | Current write-back APIs need a newer/stable release for this blob-v2 POC |

Blob v2 is validated after ingest and never silently downgraded to `large_binary`.

Current TODOs:

- `analyze` writes JSON but does not yet append scalar analysis columns back to Lance.
- `embed` appends only `audio_embedding`; similar-complaint flags are not currently computed.
- `daft_lance.merge_columns_df` is not used for embedding because the blob-v2 write-back path has correctness/compatibility issues in this POC.
- `lance-ray` can read blob bytes through `read_lance`, but the write-back path is deferred until a newer/stable release is available.

## Verified versions and runtime

Verified with the project-managed `uv.lock` / `.venv`:

| Component | Version | Notes |
|-----------|---------|-------|
| Daft | 0.7.15 | Main execution engine |
| daft-lance | 0.4.0 | Required for `read_lance`, `write_lance`, and `take_blobs` |
| Lance Python / pylance | 7.0.0 | Lance dataset, blob v2, and native ANN scanner APIs |
| lance-ray | 0.4.2 | Installed for follow-up distributed tests; not on the current embedding write-back path |
| Ray | 2.55.1 | Pulled in by `lance-ray`; Daft does not use Ray unless `USE_RAY=1` |

Default runtime:

- Daft runner: `native` (local multi-threaded).
- Set `USE_RAY=1` to run Daft-backed steps on Ray. `query` and the current Lance-native embedding write-back run locally.

Local Lance URIs are verified end-to-end. S3 Lance table write/read support is partially exercised by the underlying libraries but should be treated as a separate validation item for this POC.

## Setup

```sh
uv sync --upgrade
```

Create a `.env` file (or export directly):

```sh
MINIO_ENDPOINT=http://127.0.0.1:9000
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin
MINIO_REGION=us-east-1

DEEPSEEK_API_KEY=sk-...          # leave empty to skip LLM analysis
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat

ASR_DEVICE=cpu                   # or cuda

MIN_DURATION_S=0
MAX_DURATION_S=1800
EMBED_BACKEND=signal             # signal (128-dim) or wav2vec2

USE_RAY=0                        # set to 1 to run ingest/analyze/embed on Ray
RAY_ADDRESS=                     # Ray cluster address; empty = start/join local Ray
```

## Run

Manifest must be parquet, jsonl, or csv with `doc_id` and `s3_url` columns.
`--lance-uri` accepts both local paths and `s3://` URIs.

```sh
# Local Lance table
mmt-ingest  --manifest s3://bucket/audio/manifest.parquet \
            --lance-uri /tmp/calls.lance

# S3 Lance table
mmt-ingest  --manifest s3://bucket/audio/manifest.parquet \
            --lance-uri s3://bucket/audio/calls.lance

mmt-analyze --lance-uri s3://bucket/audio/calls.lance \
            --out-jsonl s3://bucket/audio/analysis.jsonl

mmt-embed   --lance-uri s3://bucket/audio/calls.lance

# Scalar filter
mmt-query   --lance-uri s3://bucket/audio/calls.lance \
            --where "bad_tone = true OR downgrade_related = true" \
            --top-k 5

# ANN: recordings acoustically similar to a reference
mmt-query   --lance-uri s3://bucket/audio/calls.lance \
            --query-doc-id call_001.mp3 \
            --top-k 5

# Export matching audio to local directory
mmt-query   --lance-uri s3://bucket/audio/calls.lance \
            --where "downgrade_related = true" \
            --export-audio-dir /tmp/audio_out
```

For local verification, use:

```sh
uv run python -m multimodal_toolkit.pipeline.ingest \
  --manifest s3://contacts/audio_poc/manifest.parquet \
  --lance-uri /tmp/audio_poc/calls.lance

uv run python -m multimodal_toolkit.pipeline.analyze \
  --lance-uri /tmp/audio_poc/calls.lance \
  --out-jsonl /tmp/audio_poc/analysis.json

uv run python -m multimodal_toolkit.pipeline.embed \
  --lance-uri /tmp/audio_poc/calls.lance

uv run python -m multimodal_toolkit.pipeline.query \
  --lance-uri /tmp/audio_poc/calls.lance \
  --query-doc-id call_001.mp3 \
  --top-k 5
```

Without `uv` installation, use `python -m multimodal_toolkit.pipeline.<step>` in place of `mmt-<step>`.
