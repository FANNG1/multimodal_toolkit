#!/usr/bin/env bash
# 输出连接方式；port-forward 在前台运行，使用独立终端启动它们。
set -euo pipefail

kubectl --context daft-ray --namespace daft-ray get raycluster,pods,svc,jobs
cat <<'EOF'

Ray Dashboard:
  kubectl --context daft-ray -n daft-ray port-forward service/daft-ray-head-svc 8265:8265
  open http://127.0.0.1:8265

Prometheus:
  kubectl --context daft-ray -n daft-ray port-forward service/prometheus 9090:9090
  open http://127.0.0.1:9090/targets

MinIO Console:
  kubectl --context daft-ray -n daft-ray port-forward service/minio 9001:9001
  open http://127.0.0.1:9001

Real audio smoke:
  scripts/k8s/run-audio-smoke.sh
EOF
