# multimodal_toolkit

Minimal POC for audio multimodal processing:

1. Read an S3/MinIO manifest with only doc_id and s3_url.
2. Use Daft to download audio and write a Lance table with audio_blob as Lance blob v2.
3. Analyze Lance rows with SenseVoice + DeepSeek and write JSONL plus scalar columns back to Lance.
4. Add pure acoustic signal embeddings and mark similarity to complaint seed doc_ids.
5. Query scalar fields plus embeddings/audio blobs.

Engine boundary follows the verified POC under /Users/fanng/opensource/video:

- Daft: read manifest, download S3 objects, write Lance blob v2, scalar/vector query where supported.
- Lance: materialize blob bodies with read_blobs/take_blobs and append computed columns when Daft has no stable equivalent.
- lance-ray: reserved for later scale-out add_columns/index optimization tests.

Blob v2 is a hard requirement; the code validates it after ingest and does not silently downgrade to large_binary.

## Setup

Use latest stable versions and lock them:

    uv sync --upgrade

Environment variables can be placed in .env:

    MINIO_ENDPOINT=http://127.0.0.1:9000
    MINIO_ROOT_USER=minioadmin
    MINIO_ROOT_PASSWORD=minioadmin
    MINIO_REGION=us-east-1
    DEEPSEEK_API_KEY=...
    DEEPSEEK_BASE_URL=https://api.deepseek.com
    DEEPSEEK_MODEL=deepseek-chat
    ASR_DEVICE=cpu

## Run

    python -m multimodal_toolkit.ingest --manifest s3://contacts/audio/manifest.parquet --lance-uri /tmp/audio_poc/calls.lance
    python -m multimodal_toolkit.analyze --lance-uri /tmp/audio_poc/calls.lance --out-jsonl /tmp/audio_poc/analysis.jsonl
    python -m multimodal_toolkit.embed --lance-uri /tmp/audio_poc/calls.lance --seed-doc-ids doc1,doc2 --threshold 0.80
    python -m multimodal_toolkit.query --lance-uri /tmp/audio_poc/calls.lance --where "bad_tone = true OR downgrade_related = true" --top-k 5

Manifest schema is intentionally minimal: doc_id string, s3_url string.

## Current local verification notes

On this machine, with the current Daft 0.7.15 environments:

- Daft can read the S3 manifest from MinIO.
- Daft functions.download + write_lance(blob_columns=...) can write Lance blob v2 for the audio manifest.
- Some commands may terminate without printing the final Python output even though the Lance commit succeeds; verify by opening the Lance table.
- Daft read_lance exposes blob v2 as descriptor structs. Blob body materialization uses Lance native read_blobs, matching the Guangdong image POC.
- Query supports --engine lance as a final fallback for environments where Daft read_lance cannot collect the final mixed blob/vector table reliably.

The preferred local run is:

    python -m multimodal_toolkit.ingest --manifest s3://contacts/audio_poc/manifest.parquet --lance-uri /tmp/audio_poc/calls.lance
    python -m multimodal_toolkit.analyze --lance-uri /tmp/audio_poc/calls.lance --out-jsonl /tmp/audio_poc/analysis.jsonl
    python -m multimodal_toolkit.embed --lance-uri /tmp/audio_poc/calls.lance --seed-doc-ids call_006_package_downgrade.mp3 --threshold 0.80
    python -m multimodal_toolkit.query --lance-uri /tmp/audio_poc/calls.lance --where "bad_tone = true OR downgrade_related = true" --top-k 5 --engine lance
