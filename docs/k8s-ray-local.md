# 本地 Kubernetes + Ray + Daft 完整音频环境

本环境在专用的 Minikube profile `daft-ray` 中运行真实音频链路，不读取宿主机已有的
MinIO、模型服务或 S3 数据：

```text
MinIO (cluster-local) → Daft download → SenseVoice ASR → Mock LLM → Lance Blob v2
                              │
                         KubeRay worker
                              │ :8080
                         Prometheus
```

Ray Dashboard 由 KubeRay head 提供；Prometheus 仅抓取 Ray head 和 worker 的内置
OpenMetrics 指标。本环境不安装 Grafana，也不暴露业务自定义指标。

## 前置条件

- Docker Desktop 或 OrbStack 已运行，并至少可分配 **4 CPU / 12 GiB 内存**；首次模型下载
  和完整 Python 镜像还需要若干 GiB 磁盘空间。
- 已安装 `minikube`、`kubectl`、`helm`，且能访问镜像仓库、KubeRay Helm 仓库和 PyPI。
- 首次 audio smoke 会下载 SenseVoice/VAD 模型。模型缓存位于 worker Pod 的本地卷；Pod
  重建后会重新下载，属于开发环境的有意取舍。

## 启动与访问

```bash
scripts/k8s/up.sh
scripts/k8s/status.sh
```

`up.sh` 会创建专用 Minikube profile、构建 `multimodal-toolkit:audio-local`、安装固定
版本 KubeRay operator，并部署 MinIO、Mock LLM、RayCluster 和 Prometheus。镜像完整安装
当前 `uv.lock` 的项目依赖，但声明为 CPU-only，不请求 GPU。

在独立终端执行端口转发：

```bash
kubectl --context daft-ray -n daft-ray port-forward service/daft-ray-head-svc 8265:8265
kubectl --context daft-ray -n daft-ray port-forward service/prometheus 9090:9090
```

- Ray Dashboard：<http://127.0.0.1:8265>
- Prometheus targets：<http://127.0.0.1:9090/targets>

所有 Ray target 应为 `UP`。可从以下 PromQL 起步；实际可用指标名以 target 的 `/metrics`
为准，因为会随 Ray 版本和组件变化：

```promql
ray_tasks
ray_object_store_memory
ray_node_cpu_utilization
```

## 运行真实音频 smoke

```bash
scripts/k8s/run-audio-smoke.sh
```

脚本会使用 UTC run-id，在集群内 MinIO 的 `benchmark/audio/<run-id>/` 下写入 4 条种子
音频、Parquet manifest 与独立 Lance 表，随后通过 Ray Jobs API 提交：

```text
Daft download → @daft.cls SenseVoice ASR → HTTP Mock LLM → Lance Blob v2
```

结束时脚本会验证结果表行数、所有 `status=ok` 与 `audio_blob` 的 Blob v2 格式。执行期间可在
Ray Dashboard 的 Jobs 页面查看作业和 actor；首次模型冷启动会明显比后续运行更久。

## 清理

```bash
scripts/k8s/down.sh
```

它只删除专用 profile 中的 `daft-ray` 命名空间及其 MinIO PVC，不会删除 Minikube 或
KubeRay operator。若确认不再需要整个专用集群，再手动执行：

```bash
minikube delete --profile daft-ray
```
