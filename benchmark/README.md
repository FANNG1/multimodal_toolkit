# Daft audio stability benchmark

For complete local and multi-node Ray instructions, sizing guidance, metrics,
and extension points, see the [audio benchmark guide](../docs/audio-benchmark.md).

This package exercises the real audio path without calling a real LLM:

```
MinIO -> Daft/Ray download -> SenseVoice ASR -> HTTP mock LLM -> Lance blob v2
```

It always uses the MinIO credentials and endpoint from the repository `.env`.
It never starts, stops, or deletes the configured MinIO service. Each run writes
under its own `s3://<bucket>/<prefix>/<run-id>/` prefix.

## Local smoke test

Prerequisites: the configured MinIO is running, the SenseVoice and FSMN VAD
models are cached or downloadable, and `data/audio/` contains seed audio.

```bash
uv run python -m benchmark.audio local-smoke --count 4
```

For a repeatable local performance baseline, generate 50 fixed 60-second files,
warm up the model/cache once, and run the same manifest twice:

```bash
uv run python -m benchmark.audio local-baseline
```

The comparison is written to
`.benchmarks/<run-id>/baseline-summary.{json,md}`.

Results are written to `.benchmarks/<run-id>/`, including `metadata.json`,
`summary.json`, `resources.csv`, `report.json`, `report.md`, the Daft physical
plan, and Daft event logs.

## Separate commands

```bash
uv run python -m benchmark.audio generate --profile smoke --count 4

uv run python -m benchmark.audio serve-mock --host 0.0.0.0 --port 8010 --profile standard

uv run python -m benchmark.audio run \
  --manifest s3://benchmark/audio/RUN/manifest.parquet \
  --lance-uri s3://benchmark/audio/RUN/output.lance \
  --mock-url http://MOCK_HOST:8010 \
  --ray-address auto --max-minutes 60

uv run python -m benchmark.audio submit \
  --dashboard-address http://RAY_HEAD:8265 \
  --manifest s3://benchmark/audio/RUN/manifest.parquet \
  --lance-uri s3://benchmark/audio/RUN/output.lance \
  --mock-url http://MOCK_HOST:8010 \
  --ray-address auto --max-minutes 60 --wait
```

Use `generate --profile fixed` for controlled fixed-duration data and
`--profile mixed` for a 70% short / 25% medium / 5% long distribution. Data
generation happens before the timed benchmark.
