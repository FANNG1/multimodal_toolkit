#!/usr/bin/env bash
# 生成隔离的测试音频，并以 Ray Jobs API 提交真实 Daft/SenseVoice/Mock-LMM/Lance 链路。
set -euo pipefail

kubectl_context="daft-ray"
namespace="daft-ray"
run_id="k8s-smoke-$(date -u +%Y%m%dT%H%M%SZ)"
bucket="benchmark"
prefix="audio"
head_pod="$(kubectl --context "$kubectl_context" -n "$namespace" get pod \
  -l ray.io/node-type=head -o jsonpath='{.items[0].metadata.name}')"

if [[ -z "$head_pod" ]]; then
  echo "Ray head is unavailable; run scripts/k8s/up.sh first." >&2
  exit 1
fi

common_env=(
  "MINIO_ENDPOINT=http://minio:9000"
  "MINIO_ROOT_USER=minioadmin"
  "MINIO_ROOT_PASSWORD=minioadmin"
  "MINIO_REGION=us-east-1"
  "BENCH_PROMETHEUS_PUSHGATEWAY_URL=http://pushgateway:9091"
  "NO_PROXY=minio,mock-llm,127.0.0.1,localhost"
  "no_proxy=minio,mock-llm,127.0.0.1,localhost"
)

echo "[generate] run_id=$run_id"
kubectl --context "$kubectl_context" -n "$namespace" exec "$head_pod" -c ray-head -- \
  env "${common_env[@]}" python -m benchmark.audio generate \
    --run-id "$run_id" --bucket "$bucket" --prefix "$prefix" --profile smoke --count 4 \
    --source-dir /app/data/audio

manifest="s3://$bucket/$prefix/$run_id/manifest.parquet"
output="s3://$bucket/$prefix/$run_id/output.lance"
echo "[submit] $manifest -> $output"
kubectl --context "$kubectl_context" -n "$namespace" exec "$head_pod" -c ray-head -- \
  env "${common_env[@]}" ray job submit --address http://127.0.0.1:8265 \
    --submission-id "$run_id" --working-dir /app -- \
    python -m benchmark.audio run --run-id "$run_id" --manifest "$manifest" \
      --lance-uri "$output" --mock-url http://mock-llm:8010 --ray-address auto \
      --asr-actor-cpus 2 --num-partitions 4 --max-minutes 20

kubectl --context "$kubectl_context" -n "$namespace" exec "$head_pod" -c ray-head -- \
  env "${common_env[@]}" python examples/k8s/verify_audio_smoke.py "$output" 4
echo "Audio smoke completed: $output"
