# Daft 音频稳定性与性能 Benchmark

本文档说明如何使用 `benchmark/audio` 工具验证以下链路，并从本地小数据扩展到多节点 Ray 集群：

```text
MinIO/S3 → Daft download → SenseVoice ASR → HTTP Mock LLM → Lance Blob v2
```

Benchmark 不调用真实 LLM，也不会启动或停止 MinIO。每次运行使用独立的 `run_id` 和 S3 prefix，不覆盖其他运行的数据。

## 1. 命令与产物

统一入口为：

```bash
uv run python -m benchmark.audio --help
```

| 命令 | 用途 |
|---|---|
| `generate` | 构造音频对象并写入 MinIO，同时生成 Parquet manifest |
| `serve-mock` | 启动 OpenAI-compatible Mock LLM |
| `run` | 直接连接 local/remote Ray 执行 Daft pipeline |
| `submit` | 通过 Ray Jobs API 提交同一个 pipeline |
| `report` | 从已有运行产物重建报告 |
| `local-smoke` | 一键完成 MinIO 预检、造数、本地 Ray 执行和校验 |
| `local-baseline` | 构造固定时长数据并重复运行，比较吞吐稳定性 |

每次运行在 `.benchmarks/<run_id>/` 或 `--run-dir` 指定目录生成：

| 文件 | 内容 |
|---|---|
| `metadata.json` | Git commit、依赖版本、运行参数和集群节点 |
| `summary.json` | 状态、总耗时、行数、字节数和音频时长 |
| `resources.csv` | 每 5 秒采集的节点 CPU、内存、Ray RSS 和 object store |
| `report.json` | 机器可读的吞吐、延迟、重试和内存 SLO |
| `report.md` | 适合人工阅读的摘要 |
| `plan.txt` | Daft 优化前后的执行计划 |
| `daft-events/` | Daft query、operator 和 process event log |

结果表保存在 `--lance-uri`，包含 ASR/LLM 结果、逐行耗时、状态、错误信息和原始 `audio_blob`。写入后会强制校验 Lance Blob v2。

## 2. 本地快速验证

准备条件：

- `.env` 中的 `MINIO_ENDPOINT` 和凭证可用；
- MinIO 当前已启动；benchmark 不负责启动服务；
- `data/audio/` 至少有一个支持的种子文件；
- SenseVoice 和 FSMN VAD 模型已缓存或当前环境可以下载；
- 安装了 `ffmpeg` 和 `libsndfile`。

执行真实 ASR 的四行 smoke：

```bash
uv sync --frozen
uv run python -m benchmark.audio local-smoke --count 4
```

`local-smoke` 会用随机端口启动 fast Mock LLM，只清理该 Mock 进程。它会向当前 MinIO 写入：

```text
s3://benchmark/audio/<run_id>/
├── input/
├── manifest.parquet
└── output.lance/
```

如需使用其他 bucket、prefix 或种子目录：

```bash
uv run python -m benchmark.audio local-smoke \
  --bucket perf-test \
  --prefix daft/audio \
  --source-dir /path/to/audio \
  --count 8
```

### 本地性能 Baseline

`local-smoke` 只有少量文件，仅用于验证功能。要得到有意义的本地吞吐和内存曲线，使用：

```bash
uv run python -m benchmark.audio local-baseline
```

默认行为：

- 生成 50 条固定 60 秒音频；
- 先用一条音频预热 Ray、模型和本地缓存，warm-up 不计入两轮比较；
- 两轮复用完全相同的 manifest；
- 每轮写入独立的 Lance 表，避免覆盖；
- 使用 fast Mock LLM，减少外部延迟随机性；
- 单轮最多 45 分钟；
- 输出两轮吞吐均值、首尾差异和变异系数；CV ≤ 10% 判为重复性通过。

运行目录结构：

```text
.benchmarks/<baseline-run-id>/
├── repeat-01/
├── repeat-02/
├── warmup/
├── baseline-summary.json
└── baseline-summary.md
```

常用调整：

```bash
uv run python -m benchmark.audio local-baseline \
  --count 60 \
  --duration-s 60 \
  --repeats 2 \
  --max-minutes 45
```

如果当前 Ray/模型缓存已经由其他任务预热，可传 `--skip-warmup`。常规可重复测试不建议跳过，否则首次下载/解压模型会显著抬高第一轮耗时并造成虚假的高 CV。

默认数据量约 50 音频分钟。根据此前本地约 2 audio seconds / wall second 的结果，单轮预计 20–30 分钟；实际时间仍取决于 CPU、actor 初始化和种子编码。两轮运行共享同一个 local Ray 进程，但每轮重新建立 Daft query/ASR actor，因此报告仍包含每轮 actor 初始化成本，不包含首次模型下载成本。

## 3. 构造测试数据

三种 profile 的定位如下：

| Profile | 行为 | 适用场景 |
|---|---|---|
| `smoke` | 直接复用种子文件内容，但每行使用唯一对象 key | 功能验证、依赖验证 |
| `fixed` | 循环/裁剪种子并转为 16kHz 单声道 MP3/WAV | 参数间公平对比 |
| `mixed` | 70% 15–90s、25% 90–300s、5% 300–1800s；80% MP3、20% WAV | 调度、长尾和内存稳定性 |

固定 60 秒、1000 条：

```bash
RUN_ID=audio-fixed-$(date +%Y%m%d-%H%M%S)
uv run python -m benchmark.audio generate \
  --run-id "$RUN_ID" \
  --profile fixed \
  --duration-s 60 \
  --count 1000 \
  --bucket benchmark \
  --prefix audio
```

混合时长数据：

```bash
uv run python -m benchmark.audio generate \
  --run-id "$RUN_ID" \
  --profile mixed \
  --count 10000 \
  --seed 20260715
```

同一个 seed、种子目录和参数会得到相同的时长/编码选择。所有音频会使用唯一 S3 key，避免因重复 URL 缓存而低估 I/O 成本。造数时间不计入 benchmark 运行时间。

## 4. 扩展到 4-worker Ray 集群

基准拓扑：一个不承担 ASR 的 Ray head，加 4 个 worker；每个 worker 为 8 CPU、24GB 内存。启动 head 时应将可调度 CPU 设为 0，建议每个 worker 预留约 6GB Ray object store，并配置本地 SSD spill 目录。

```bash
ray start --head --num-cpus=0
```

### 4.1 集群前置检查

所有节点必须使用相同的 Python/系统环境，并具备：

- 本项目及 `uv.lock` 中的依赖；
- `ffmpeg`、`libsndfile`；
- 相同的 SenseVoice/VAD 模型缓存；
- 可写的临时目录和 Ray spill 目录；
- 到 MinIO 和 Mock LLM 的网络路由。

多节点时不能使用 `127.0.0.1` 作为 MinIO 或 Mock 地址。将 `.env` 中的端点改为所有 worker 可访问的地址，并配置 `NO_PROXY/no_proxy`：

```bash
MINIO_ENDPOINT=http://minio.internal:9000
MINIO_ROOT_USER=...
MINIO_ROOT_PASSWORD=...
MINIO_REGION=us-east-1
NO_PROXY=minio.internal,mock-llm.internal,127.0.0.1,localhost
no_proxy=minio.internal,mock-llm.internal,127.0.0.1,localhost
ASR_DEVICE=cpu
```

逐节点验证：

```bash
curl -f http://minio.internal:9000/minio/health/live
curl -f http://mock-llm.internal:8010/health
ray status
```

`submit` 会打包项目源码并传递 benchmark 所需环境变量，但不会安装系统库或下载模型。生产式压测应使用预构建且一致的 worker 镜像。

### 4.2 启动 Mock LLM

在所有 worker 可访问的主机运行：

```bash
uv run python -m benchmark.audio serve-mock \
  --host 0.0.0.0 \
  --port 8010 \
  --profile standard
```

`standard` profile 的中位延迟约 2 秒、P95 约 8 秒、响应约 4KB，并确定性注入少量首请求 429/500；后续 retry 可恢复。Mock 自身也可能成为瓶颈，应独立观察其 CPU、连接数和错误率。

### 4.3 提交第一轮安全基线

4×8 CPU 集群先使用每个 ASR actor 4 CPU、最多 4 个 actor、batch size 1。ASR actor 使用 Ray 的 `SPREAD` 调度策略，但这只是调度倾向，不是严格的“一节点一个 actor”约束；必须从 Ray Dashboard 和逐节点 RSS 确认实际放置。`--num-partitions 64` 对应总 CPU 的两倍，确保小 manifest 能在 download/ASR 前充分拆分：

```bash
uv run python -m benchmark.audio submit \
  --dashboard-address http://ray-head.internal:8265 \
  --run-id "$RUN_ID" \
  --manifest "s3://benchmark/audio/$RUN_ID/manifest.parquet" \
  --lance-uri "s3://benchmark/audio/$RUN_ID/output.lance" \
  --mock-url http://mock-llm.internal:8010 \
  --ray-address auto \
  --max-minutes 60 \
  --num-partitions 64 \
  --asr-actor-cpus 4 \
  --asr-actor-concurrency 4 \
  --asr-batch-size 1 \
  --llm-concurrency 32 \
  --wait
```

不传 `--num-partitions` 时会自动使用集群 CPU 数的两倍。显式参数更适合可重复的横向比较。

`--asr-actor-concurrency` 控制可并行的同步 ASR actor 数；`--llm-concurrency` 是单个 Daft 异步 LLM actor 的最大协程数，不代表全集群 HTTP 请求总数。总并发必须结合 Ray actor 数量和 Mock 端连接数实测。

Ray Jobs 的默认本地报告位于 job 工作目录，可能随 runtime environment 清理。集群验收应把 `--run-dir` 指向 Ray head 可写的持久卷，例如：

```bash
--run-dir "/mnt/benchmark/$RUN_ID"
```

结果 Lance 始终保存在 MinIO；报告目录只保存指标和诊断信息。

## 5. 一小时迭代与调参顺序

ASR 工作量更接近“音频时长”而不是文件字节数。不要直接假设固定 GB 数能在一小时完成。

1. 用 100–500 条代表性数据跑 10–15 分钟校准。
2. 从 `report.json` 读取 `audio_seconds_per_wall_second = R`。
3. 正式批次目标音频秒数取 `R × 2700`，即约 45 分钟有效负载，给冷启动和 Lance commit 留 15 分钟。
4. `mixed` profile 的理论平均时长约 138 秒，可用 `目标音频秒数 ÷ 138` 粗估 `--count`。
5. 同一配置至少重复两次；吞吐差异超过 10% 时先排查 MinIO、Mock 和节点噪声。

建议一次只改变一个参数：

1. 固定 batch 1，比较 ASR actor concurrency 4 与 8；只有内存 SLO 通过时才提高。
2. 固定 actor 数，比较 ASR batch size 1 与 2。
3. 比较 `DAFT_DEFAULT_MORSEL_SIZE=8/16/32`；大音频优先从 8 开始。
4. 最后调整 `--llm-concurrency`、S3 连接数和 partition 数。

更多 Daft 参数含义见 [Daft tuning](daft-tuning.md)。

当前 `generate` 使用单进程串行 ffmpeg 和串行上传，适合提前准备小到中等规模数据，不适合临时生成 TB 级数据。大规模造数应在正式计时前完成；需要 TB 级数据时可沿用同一 manifest schema，另行实现并行生成器。

## 6. 报告与验收

关键指标：

- `files_per_s`：小文件调度能力；
- `mb_per_s`：S3/Lance 字节吞吐；
- `audio_seconds_per_wall_second`：ASR 核心吞吐；
- `asr_ms`、`llm_ms` P50/P95/P99：只统计实际执行过对应阶段的行，跳过的 0ms 不进入分位数；
- `ray_process_rss_*`：Ray/Daft 相关进程内存；
- `resources.nodes.*`：逐节点总内存、增长斜率和 SLO；
- `status_counts`、`llm_retry_rows`：正确性与恢复情况。

默认内存 SLO：逐节点 P95 不超过总内存 80%，峰值不超过 90%，增长斜率小于 0.5GB/小时。节点总内存包含同机其他进程，容器中还可能反映宿主机而非 cgroup；Ray RSS 也是按进程命令行近似归因。因此这些值是诊断性指标，正式结果应来自隔离 worker，并结合 Ray Dashboard/Prometheus 复核。当前工具不采集磁盘 spill 字节数。

一次运行至少满足：

- query 在 `--max-minutes` 内完成；该参数是 driver 侧 best-effort deadline，集群硬时限还应使用外部 watchdog 或 `ray job stop`；
- 无 OOM、无限等待或未恢复 Ray task failure；
- manifest 行数与 Lance 行数一致，`doc_id` 唯一；
- 每行均为 `ok` 或明确的失败状态；
- `audio_blob` 通过 Lance Blob v2 校验；
- 两次相同配置吞吐变异系数不超过 10%。

## 7. 扩展开发

主要模块：

| 模块 | 扩展点 |
|---|---|
| `data.py` | 新的数据分布、编码组合、坏文件和 TTS 生成器 |
| `mock_llm.py` | 新延迟模型、限流、超时和响应大小 |
| `pipeline.py` | 新 Daft UDF、资源声明、输出列和失败语义 |
| `metrics.py` | Prometheus、GPU、磁盘 spill 和网络指标 |
| `report.py` | 新 SLO、横向对比和趋势分析 |
| `submit.py` | Kubernetes、容器镜像和 Ray runtime environment |

新增 profile 或指标时，应同时补充 `tests/benchmark/`，保持默认 pytest 不依赖 MinIO/Ray/模型；真实端到端测试继续使用 `benchmark_e2e` marker。

当前限制：pipeline 只接受 Parquet manifest，只覆盖 CPU SenseVoice 和音频；尚未在真实 4-worker 集群验收，不包含 GPU、视频/图片、真实 LLM 或主动 worker 故障注入。每轮必须使用唯一 `run_id`/Lance URI，因为写入模式为 `overwrite`。
