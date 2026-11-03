#!/usr/bin/env bash
# 输出集群状态和本地 NodePort 访问地址。
set -euo pipefail

kubectl --context daft-ray --namespace daft-ray get raycluster,pods,svc,jobs
minikube_ip="$(minikube ip --profile daft-ray)"
cat <<EOF

Ray Dashboard:
  http://${minikube_ip}:30265

Prometheus:
  kubectl --context daft-ray -n daft-ray port-forward service/prometheus 9090:9090
  open http://127.0.0.1:9090/targets

Grafana:
  http://${minikube_ip}:30300

MinIO Console:
  kubectl --context daft-ray -n daft-ray port-forward service/minio 9001:9001
  open http://127.0.0.1:9001

Real audio smoke:
  scripts/k8s/run-audio-smoke.sh
EOF
