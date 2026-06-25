# multimodal_toolkit

Audio call-centre analysis POC: ingest recordings from S3, transcribe with SenseVoice, analyse with DeepSeek, embed acoustically, and query by filter or nearest-neighbour — all in a Daft-native pipeline backed by Lance blob v2.

## Pipeline

| Step | Module | What it does |
|------|--------|--------------|
| 1 ingest | `ingest` | Read manifest → Daft S3 download → write Lance blob v2 table |
| 2 analyze | `analyze` | `take_blobs` → duration filter → SenseVoice ASR → PII redaction → DeepSeek LLM → write JSONL + append scalar columns |
| 3 embed | `embed` | `take_blobs` → acoustic signal embedding (128-dim RMS+ZCR) → cosine similarity to seed complaints → append vector + flag columns |
| 4 query | `query` | Scalar filter via Daft or ANN via Lance native |

## Engine decisions

| Engine | Used for | Reason |
|--------|----------|--------|
| **Daft** | manifest read, S3 download, Lance write, blob materialization (`daft_lance.take_blobs`), scalar query, LLM/ASR pipeline | Primary compute engine |
| **Lance native** | ANN query (`scanner(nearest=...)`) | `daft.read_lance` hides `_distance`; Lance native exposes it for ranking |
| **lance-ray** | `add_columns` after analyze/embed | Primary engine for appending computed columns; Lance native is the fallback |

Blob v2 is validated after ingest and never silently downgraded to `large_binary`.

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
```

## Run

Manifest must be parquet, jsonl, or csv with `doc_id` and `s3_url` columns.

```sh
mmt-ingest  --manifest s3://bucket/audio/manifest.parquet \
            --lance-uri /tmp/calls.lance

mmt-analyze --lance-uri /tmp/calls.lance \
            --out-jsonl /tmp/analysis.jsonl

mmt-embed   --lance-uri /tmp/calls.lance \
            --seed-doc-ids call_001.mp3,call_002.mp3 \
            --threshold 0.80

# Scalar filter
mmt-query   --lance-uri /tmp/calls.lance \
            --where "bad_tone = true OR downgrade_related = true" \
            --top-k 5

# ANN: recordings acoustically similar to a reference
mmt-query   --lance-uri /tmp/calls.lance \
            --query-doc-id call_001.mp3 \
            --top-k 5

# Export matching audio to local directory
mmt-query   --lance-uri /tmp/calls.lance \
            --where "downgrade_related = true" \
            --export-audio-dir /tmp/audio_out
```

Without `uv` installation, use `python -m multimodal_toolkit.<step>` in place of `mmt-<step>`.
