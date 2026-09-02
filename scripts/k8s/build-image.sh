#!/usr/bin/env bash
# 构建与本地源码一致的 Ray runtime 镜像，并载入 Minikube 的节点镜像缓存。
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
image="multimodal-toolkit:audio-local"

docker build --quiet --file "$project_root/deploy/k8s/Dockerfile.audio" --tag "$image" "$project_root"
minikube image load "$image" --profile daft-ray
