# Daft 音频稳定性基准测试

完整的本地与多节点 Ray 操作说明、容量规划、指标口径和扩展点，见
[音频基准测试指南](../docs/audio-benchmark.md)。

本 package 在不调用真实 LLM 的前提下跑通完整的音频链路：

```
MinIO -> Daft/Ray 下载 -> SenseVoice ASR -> HTTP mock LLM -> Lance blob v2
```

它始终使用仓库 `.env` 中的 MinIO 凭据与 endpoint，绝不会启动、停止或删除已配置的
MinIO 服务。每次运行写入各自独立的 `s3://<bucket>/<prefix>/<run-id>/` 前缀下。

## 本地冒烟测试

前置条件：配置的 MinIO 已在运行，SenseVoice 与 FSMN VAD 模型已缓存或可下载，
且 `data/audio/` 中有种子音频。

```bash
uv run python -m benchmark.audio local-smoke --count 4
```

如果需要可重复的本地性能 baseline —— 生成 50 条固定 60 秒音频，预热一次模型/缓存，
再用同一份 manifest 跑两轮：

```bash
uv run python -m benchmark.audio local-baseline
```

比较结果写入 `.benchmarks/<run-id>/baseline-summary.{json,md}`。

运行产物写入 `.benchmarks/<run-id>/`，包含 `metadata.json`、`summary.json`、
`resources.csv`、`report.json`、`report.md`、Daft 物理执行计划以及 Daft 事件日志。

## 分步命令

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

`generate --profile fixed` 用于生成受控的固定时长数据，`--profile mixed` 生成
70% 短 / 25% 中 / 5% 长 的分布。数据生成发生在计时的基准测试之前。
