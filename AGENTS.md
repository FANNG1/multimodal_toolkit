# multimodal_toolkit — Agent 指南

## 语言约定（重要）

本项目**面向中文读者**：

- **代码注释用中文**，且要详细：模块 docstring 说明该文件在流水线中的角色和设计取舍；关键字段、阈值、非显然的逻辑都要有中文解释（参考 `multimodal_toolkit/image/udfs.py` 的注释密度）。
- **PR 的标题和描述用中文**，commit message 可保留英文。
- README 面向使用者，中英混排可接受；新增使用文档优先中文。

## 项目概览

多模态（音频/图片）数据处理工具箱：manifest → 分析（ASR/人脸/清晰度/LLM 打标）→ Lance 资产表（blob v2）→ 索引 → 查询。计算引擎是 Daft（native 或 Ray runner），存储是 Lance on S3/MinIO。

每个媒体类型是独立的垂直管线，各自 5 个 Stage（analyze / ingest / index / query / manage）：

```
multimodal_toolkit/
  config.py            # 共享配置：env 读取辅助函数、S3、Daft 执行参数、Lance 写出参数
  storage/io.py        # 共享 infra：runner 配置、IOConfig、manifest 读取、分区辅助
  storage/blob.py      # lance blob v2 校验
  audio/               # 音频垂直：ASR(SenseVoice) + DeepSeek 打标 + 声学 embedding
  image/               # 图片垂直：SCRFD 人脸 + Laplacian 清晰度 + CLIP embedding
    workflow/          # 每个 Stage 一个入口文件（analyze.py / ingest.py / ...）
tests/                 # pytest；按 audio/image/storage/workflow 分目录
```

## 硬性架构约定

1. **媒体隔离**：audio 和 image 各自拥有独立的 package 和 workflow 入口，**不要**把共享 Stage 参数化成"通用管线"。可共享的只有 `storage/` 层 infra 和 `config.py` 的辅助函数。
2. **数据 API 优先级**：所有数据操作优先用 **Daft**；Daft 做不到再用 **lance-ray**；**pylance 直接调用是最后手段**（目前锁在 <8，blob compaction 被禁用）。
3. **行不丢弃**：analyze 阶段 manifest 里每个条目对应输出一行，失败用 `status` 列标记（`download_failed` / `decode_failed` / ...），分数结论置 null。合规场景必须能区分"内容有问题"和"根本没处理"。
4. **配置只走环境变量**，经 `config.py` 的 `env_int`/`env_bool`/`env_choice` 等辅助函数读取。注意：Daft 原生只读少数 `DAFT_*` 环境变量，新增 Daft 执行参数必须在 `storage/io.py::configure_daft_runner()` 里显式传给 `daft.set_execution_config()`，否则是死配置；S3 参数同理要传进 `daft_io_config()` 的 `S3Config`。
5. **写出受控**：`write_lance` 必须传 `LANCE_MAX_ROWS_PER_FILE` / `LANCE_MAX_BYTES_PER_FILE`；JSONL 写出前用 `coalesce_for_write()` 收敛分区，避免小文件。
6. **UDF 模式**：模型类 UDF 用 `@daft.cls(cpus=..., max_retries=2)`，模型在 `__init__` 里加载一次（每 worker 进程一次，不是每行）；batch 方法里逐行处理并保持 null 对齐。

## 常用命令

```sh
uv sync --upgrade                  # 安装依赖（uv 管理）
.venv/bin/python -m pytest tests   # 全量测试
USE_RAY=1 ...                      # 切换 Ray runner（默认 native）
```

- 本地依赖 MinIO（默认 `http://127.0.0.1:9000`，minioadmin/minioadmin），配置见 `.env` / README Setup 小节。
- venv 里的 lance-ray 是本地 dev checkout（版本号 0.4.2 与 PyPI 同名但内容不同），不要"顺手升级"它。

## 提交与 PR

- 不要直接提交到 `main`，从 `main` 拉 feature 分支。
- PR 描述用中文，结构：背景（为什么改）→ 改动（分类列出）→ 验证（测试结果、手工验证步骤）。
- 提交前跑全量 `pytest`；涉及 Daft 执行配置的改动，要手工确认 `daft.context.get_context().daft_execution_config` 里参数确实生效。
