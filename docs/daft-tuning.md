# Daft 参数调优指南（本项目版）

面向对 Daft 原理不太熟悉的同学。目标：大规模在 Ray 上跑 analyze/ingest 时，知道每个参数是干什么的、什么时候该调、出了问题往哪个方向查。

---

## 一、先花三分钟理解 Daft 是怎么跑我们的任务的

以图片 analyze 为例，我们写的代码逻辑上是一条流水线：

```
读 manifest → 下载图片 → 人脸/清晰度模型 → 规则打分 → 写 JSONL/Lance
```

Daft 不会一行一行地执行它，理解下面四个概念就够用了：

**1. Runner（在哪跑）**
Daft 有两种执行方式：`native`（单机多线程，调试用）和 `ray`（分布式，多机多进程）。我们用 `USE_RAY=1` 切换。**单机调试用 native，正式大规模跑用 ray**。

**2. Partition（活儿怎么分）**
Daft 把数据切成若干个"分区"，每个分区是一份独立的活儿，可以发给集群里任何一台机器。可以理解成：把 100 万张图的清单撕成 N 摞，每摞发给一个工人。
**关键点**：分区数决定了并行度上限。如果只有 1 个分区，就算集群有 100 台机器，也只有 1 个工人在干活——而这正是我们踩过的坑：manifest 文件只有几 MB，Daft 按文件大小切分区时根本切不动，导致整条链路挤在一个任务里。所以我们在下载之前显式切分区（`ANALYZE_NUM_PARTITIONS`）。

**3. Morsel（一口吃多少）**
在每个分区内部，Daft 也不是把整摞数据一次性塞给下游，而是切成更小的"一口"（morsel，默认 131072 行）流水线式传递。
**关键点**：我们的行里带着图片/音频的原始字节（一行可能几百 KB 到几 MB），如果按默认的 13 万行一口，一口就是几十 GB——直接 OOM。所以我们把 `DAFT_DEFAULT_MORSEL_SIZE` 压到 32。**行很"胖"的时候，一口要小**。

**4. Actor UDF（模型怎么加载）**
`ImageQualityUDF`、`AsrUDF` 这类带模型的 UDF 用的是 `@daft.cls`：Daft 会启动若干个常驻进程（actor），每个进程在启动时加载一次模型，然后源源不断地处理发来的数据。模型只加载一次而不是每行一次，这是快的关键；代价是 actor 启动慢（要加载甚至下载模型），所以有"就绪超时"这个参数。

一个健康的大规模作业，应该是三件事在流水线上同时发生：

```
CPU/网络 在下载下一批图片
模型     在处理当前批
写出线程  在落盘上一批结果
```

调优的本质就是让这条流水线不空转（并行度够）、不堵死（内存不炸、外部服务不被打爆）。

---

## 二、重要前提：参数是怎么生效的

**Daft 本身只认识极少数环境变量**。我们项目里所有 `DAFT_*`、`S3_*` 参数，都是由 `config.py` 读环境变量、再由 `storage/io.py::configure_daft_runner()` 在每个 workflow 启动时显式传给 Daft 的。

所以：

- 调参数 = 改 `.env` 或 export 环境变量，然后重跑命令，不需要改代码；
- 如果要**新增**一个 Daft 参数，光定义环境变量没用，必须在 `configure_daft_runner()`（执行参数）或 `daft_io_config()`（S3 参数）里接一下，否则就是死配置。

---

## 三、参数速查表

### 3.1 并行度（先调这个，收益最大）

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `USE_RAY` | `0` | `1` = 用 Ray 分布式跑。大规模必开。 |
| `RAY_ADDRESS` | 空 | 空 = 本地起/加入 Ray；连已有集群填 `ray://...`。 |
| `ANALYZE_NUM_PARTITIONS` | `auto` | analyze 阶段把 manifest 切成几摞活儿。auto = Ray 模式下取 2×集群 CPU 总核数；native 模式不切。 |

怎么定分区数：**够用就好，不是越多越好**。

- 太少：机器吃不满；单个任务太大，失败重跑代价高。
- 太多：调度开销大、Ray 元数据多、JSONL 输出小文件多（虽然写出前会自动收敛到约 N/8 个文件）。
- 经验值：2~4 × 集群总 CPU 核数。auto 的推导（2×）通常不用动。

### 3.2 内存安全

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `DAFT_DEFAULT_MORSEL_SIZE` | `32` | 每"一口"多少行。行里带图片/音频字节，所以要小。 |

估算方法：`morsel 行数 × 平均单条字节数` 就是单口内存量。图片平均 1MB 时，32 行 ≈ 32MB，安全。频繁 OOM 时优先降它（16 甚至 8），而不是加机器内存。

### 3.3 外部 LLM 服务

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `DEEPSEEK_CONCURRENCY` | `8` | 每个并行任务内同时发几个 LLM 请求。 |

**最容易被忽略的一点：总并发 = 这个值 × 并行任务数。** 比如 20 个分区 × 8 并发 = 最多 160 个同时在途的请求。调大之前先算一下供应商的限流额度（RPM/TPM），被限流后重试风暴反而更慢。LLM 单条失败不会中断作业（`on_error="ignore"` + 下游 null 兜底），失败行对应字段为默认值。

### 3.4 S3 / MinIO

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `S3_MAX_CONNECTIONS` | `8` | 每个 IO 线程的连接数。下载慢且 MinIO 不忙时可加。 |
| `S3_NUM_TRIES` | `5` | 单个请求最大尝试次数。 |
| `S3_RETRY_INITIAL_BACKOFF_MS` | `1000` | 重试退避起点。 |
| `S3_CONNECT_TIMEOUT_MS` | `10000` | 连接超时。 |
| `S3_READ_TIMEOUT_MS` | `60000` | 读超时。大文件/慢网络可加大。 |
| `S3_RETRY_MODE` | `adaptive` | 自适应退避，一般不动。 |

注意方向：如果 MinIO 已经在报 503/SlowDown，**该降的是并行度（分区数）**，加连接数和重试只会火上浇油。

### 3.5 Actor / 稳定性

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `DAFT_ACTOR_UDF_READY_TIMEOUT` | `600` | 等 actor 就绪的秒数。几十个 actor 同时冷启动、第一次跑还要下载模型时，300s 会集体超时，所以默认给到 600。 |

模型 UDF 都带 `max_retries=2`（写在代码里，非环境变量）：worker 被抢占或偶发失败时任务自动重试，不会整个作业挂掉。**大规模跑之前，把 insightface/SenseVoice 模型预置到镜像或共享盘**，比调超时更治本。

### 3.6 写出布局

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `LANCE_MAX_ROWS_PER_FILE` | `100000` | 单个 Lance 数据文件最大行数。 |
| `LANCE_MAX_BYTES_PER_FILE` | `512MB` | 单个 Lance 数据文件最大字节数。 |
| `DAFT_JSON_TARGET_FILESIZE` | `128MB` | JSONL 单文件目标大小。 |

为什么要管：碎文件会拖慢后续所有读取，而我们的 pylance 7 禁用了 blob compaction，写坏了没有事后补救，只能在写入侧控制。默认值一般不用动。

### 3.7 观测（长任务建议必开）

| 参数 | 说明 |
| --- | --- |
| `DAFT_EVENT_LOG_ENABLED=1` | 开启 Daft 事件日志（Daft 原生支持，无需项目代码）。 |
| `DAFT_EVENT_LOG_DIR=...` | 日志目录，放共享盘或 S3 挂载路径。 |

跑几个小时的作业中途出问题，靠事件日志能定位"失败集中在哪些文件、哪个阶段慢"，比加 print 重跑便宜得多。

---

## 四、按症状排查

| 症状 | 大概率原因 | 调整方向 |
| --- | --- | --- |
| 集群很多机器闲着 | 分区太少 | 调大 `ANALYZE_NUM_PARTITIONS`；确认真的在 Ray 模式（`USE_RAY=1`） |
| worker OOM / Ray object store 报满 | morsel 太大，在途字节太多 | 调小 `DAFT_DEFAULT_MORSEL_SIZE`；其次减少分区并行度 |
| LLM 阶段吞吐上不去 | 并发太低 | 加 `DEEPSEEK_CONCURRENCY`，但先算限流预算 |
| LLM 大量 429/超时 | 总并发超了供应商限额 | 降 `DEEPSEEK_CONCURRENCY` 或分区数 |
| 下载阶段慢、模型阶段闲 | S3 吞吐不足 | 加 `S3_MAX_CONNECTIONS`；检查 MinIO 自身负载 |
| MinIO 报 503/SlowDown | 请求压力过大 | **降**分区数，别加连接和重试 |
| 作业启动时卡很久然后报 actor 超时 | 模型冷启动/下载慢 | 预置模型；加 `DAFT_ACTOR_UDF_READY_TIMEOUT` |
| 输出一堆小文件 | 分区太碎 | JSONL 会自动收敛（约 N/8）；Lance 检查 `LANCE_MAX_*` 是否被改小 |
| 个别坏样本导致任务失败 | 不应该发生 | analyze 设计上不丢行（status 列标记失败）；真发生了是 bug，带上事件日志报修 |

---

## 五、推荐调优顺序

1. **确认 runner**：正式跑 `USE_RAY=1`，调试用 native。
2. **确认并行度**：看 Ray dashboard，任务数是否 ≈ 分区数、机器是否都在干活。
3. **确认不 OOM**：小批量数据试跑，观察 worker 内存；不稳就降 morsel。
4. **再提吞吐**：逐步加 LLM 并发 / S3 连接，每次只动一个参数，观察一轮再动下一个。
5. **开事件日志**，然后再放全量数据。

一个大规模跑的 `.env` 起点（在 README Setup 的基础上追加）：

```sh
USE_RAY=1
RAY_ADDRESS=ray://<head-node>:10001
ANALYZE_NUM_PARTITIONS=auto
DAFT_DEFAULT_MORSEL_SIZE=32
DEEPSEEK_CONCURRENCY=8
DAFT_EVENT_LOG_ENABLED=1
DAFT_EVENT_LOG_DIR=/shared/daft-events
```

记住总原则：**先找到当前瓶颈在哪一段（下载 / 模型 / LLM / 写出），只调那一段的参数**。盲目把所有并发都拉满，通常只是把瓶颈从一个地方推到另一个地方，还顺便把稳定性搞没了。
