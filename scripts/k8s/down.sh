#!/usr/bin/env bash
# 仅删除专用 daft-ray profile 中本项目命名空间的资源；不会删除 Minikube 或 operator。
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
kubectl --context daft-ray delete namespace daft-ray --ignore-not-found
