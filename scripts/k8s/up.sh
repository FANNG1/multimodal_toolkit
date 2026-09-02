#!/usr/bin/env bash
# 启动专用 Minikube、完整音频 Ray 集群、MinIO、Mock LLM 与 Prometheus。
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
kubectl_context="daft-ray"

minikube start --profile "$kubectl_context" --driver=docker --cpus=4 --memory=12288
"$project_root/scripts/k8s/build-image.sh"

helm repo add kuberay https://ray-project.github.io/kuberay-helm/ --force-update
helm repo update kuberay
helm upgrade --install kuberay-operator kuberay/kuberay-operator \
  --kube-context "$kubectl_context" \
  --namespace kuberay-system --create-namespace --version 1.4.1

kubectl --context "$kubectl_context" apply -f "$project_root/deploy/k8s/namespace.yaml"
kubectl --context "$kubectl_context" apply -f "$project_root/deploy/k8s/minio.yaml"
kubectl --context "$kubectl_context" apply -f "$project_root/deploy/k8s/prometheus.yaml"
kubectl --context "$kubectl_context" apply -f "$project_root/deploy/k8s/mock-llm.yaml"

kubectl --context "$kubectl_context" wait --namespace kuberay-system \
  --for=condition=available deployment/kuberay-operator --timeout=5m
kubectl --context "$kubectl_context" rollout status --namespace daft-ray deployment/minio --timeout=5m
kubectl --context "$kubectl_context" wait --namespace daft-ray \
  --for=condition=complete job/minio-init --timeout=5m

kubectl --context "$kubectl_context" apply -f "$project_root/deploy/k8s/raycluster.yaml"
kubectl --context "$kubectl_context" apply -f "$project_root/deploy/k8s/ray-metrics-service.yaml"

kubectl wait --context "$kubectl_context" --namespace daft-ray \
  --for=condition=ready pod -l ray.io/node-type=head --timeout=5m
kubectl wait --context "$kubectl_context" --namespace daft-ray \
  --for=condition=ready pod -l ray.io/node-type=worker --timeout=5m
kubectl rollout status --context "$kubectl_context" --namespace daft-ray deployment/prometheus --timeout=5m
kubectl rollout status --context "$kubectl_context" --namespace daft-ray deployment/mock-llm --timeout=5m

echo "Environment ready. Run scripts/k8s/run-audio-smoke.sh for the real audio pipeline."
