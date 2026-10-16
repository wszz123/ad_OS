# 统一 Memory Adapter 与 LangGraph Store 接入

## 本轮目标

在不修改 Planner/Retriever/Executor/Summarizer 的情况下，更换共享记忆存储实现。
固定任务、协议模式、模型、生成预算、纠错策略、源语料和状态/checkpoint 后端，
单独验证 Memory Adapter 的可替换性。

已经接入的后端：

| 后端 | 实现 | 生命周期 |
| --- | --- | --- |
| `sqlite` | `MemoryAdapter`，标准库 SQLite 文件 | 写入提交后可跨进程重开 |
| `langgraph` | `LangGraphMemoryAdapter` 包装官方 `InMemoryStore` | 同一 Store 实例内跨 task/thread；新进程会丢失 |

这是对实际 LangGraph 存储 API 的集成，不是仅实现一个同名假后端。
尚未接入 Mem0、Letta、PostgreSQL 或其他独立记忆服务，不能据此声称这些后端已验证。

## 接口与模块

`MemoryBackend` 提供：

```python
put(item) -> memory_id
get(memory_id) -> MemoryItem | None
get_metadata(memory_id) -> MemoryItem | None  # embedding=None
search(query, tags=None, limit=5) -> list[MemorySearchResult]
search_vector(vector, tags=None, limit=5) -> list[MemorySearchResult]
items() -> list[MemoryItem]
count() -> int
close()
```

- `coop/memory.py`：统一记忆数据结构、SQLite 存储，以及公共关键词/精确余弦排序规则。
- `coop/memory_backends.py`：LangGraph 适配器、后端工厂、公开操作计时装饰器。
- `tools/run_cost_workload.py`：新增 `--memory-backend sqlite|langgraph`；运行器不再读取记忆后端的私有 SQL 连接。
- `tools/run_memory_backend_comparison.py`：两个后端的独立 trial、反序运行、输出及成本一致性检查。

LangGraph 数据以 `namespace + memory_id` 写入 `BaseStore.put()`，记忆 ID、来源 Agent、
创建时间、主题、摘要、标签、来源链和向量均保留为 JSON 文档。
Store 的内部创建时间与记忆的 `created_at` 是不同字段，后者按统一记忆协议保存。
取出数据时脱离 Store 的可变对象引用，防止调用方直接改坏底层记录。

`BaseStore.search()` 是前缀匹配且默认只返回十条。本适配器显式分页，并检查完整
namespace 相等，防止混入子命名空间。按 memory_id 排序使并列分数不受插入顺序影响。
`close()` 不删除共享 Store 的其他 namespace，也不负责关闭调用方注入的 Store。

两个后端共用关键词、标签和精确余弦排序；`search_vector()` 直接消费已恢复的向量，
不重新编码文本。当前是 Adapter 级扫描，不是 LangGraph 原生语义索引或 ANN。
后续换原生索引应作为独立功能消融，不能同时改变排序与存储后声称只比较后端。

## 实际运行结果

报告：`results/memory_backend_comparison.json`。
两次 trial，各后端十轮，第二次反转执行顺序，任务记忆每次从空库开始。
使用Qwen3-8B `cuda:1`、Qwen3-Embedding-4B CPU、八个线程；权重bfloat16、向量输出float32，见runtime。

| 每十轮，两次平均 | SQLite | LangGraph InMemoryStore |
| --- | ---: | ---: |
| 完成/严格质量通过 | 10/10 | 10/10 |
| 模型调用 | 31 | 31 |
| LLM 输入/输出 token | 11406 / 1641 | 11406 / 1641 |
| 消息 token | 19697 | 19697 |
| 接受事实复用 | 4 | 4 |
| 语料检索 | 6 | 6 |
| 新记忆记录 | 10 | 10 |
| 任务总时间，秒 | 57.503 | 56.563 |
| 10 次记忆写入总时间，秒 | 1.095824 | 0.007225 |
| 10 次向量查询总时间，秒 | 0.007780 | 0.014319 |
| 重启后自动保留任务记忆 | 是 | 否 |

每轮的公式、工具结果、答案、质量、复用 ID/来源和纠错尝试，以及关键指标均一致。
所有检查通过，总计 **40/40 次质量通过**。因此外部 Store 替换未改变 Agent 行为。
RAM 写入与 SQLite 提交的持久语义不同；上述小样本操作/任务时间不支持通用速度结论。
操作计时覆盖 Adapter 编解码/枚举/排名与存储 API，不能当成底层数据库纯 I/O 时间。
统计快照导出不计入运行期操作计时。

## 复现

无需新增依赖，使用 `coop/requirements.txt` 已固定的 LangGraph 1.0.10。

```bash
conda run -n latentmas python -m unittest discover -s tests -v

# 固定为协议+向量+记忆，有界纠错，SQLite 状态和 checkpoint
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 conda run --no-capture-output -n latentmas \
  python -m tools.run_memory_backend_comparison \
  --device cuda:1 --embedding-device cpu --max-new-tokens 192 \
  --max-plan-repairs 2 --threshold 0.55 --repeat 2 \
  --report results/memory_backend_comparison.json

# 单独选择 LangGraph 记忆后端
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 conda run --no-capture-output -n latentmas \
  python -m tools.run_cost_workload --memory-backend langgraph \
  --mode protocol_vector_memory --repair-plan --durable \
  --device cuda:1 --embedding-device cpu --report results/cost_langgraph_memory.json
```

每个 trial/backend 的工件保留 `trace.json`、`memory_snapshot.json`、语料、向量状态和
checkpoint。SQLite 另保留 `memory.sqlite`；LangGraph RAM 后端没有该文件。
快照是实验审计导出，不实现自动恢复。`--durable` 只保证状态与 checkpoint 持久化，
不能把 process-local Store 变为持久记忆。原来的 SIGKILL 恢复验收仍使用 SQLite 记忆。

## 注入其他 Store

```python
from coop.memory_backends import LangGraphMemoryAdapter

memory = LangGraphMemoryAdapter(
    store=my_sync_langgraph_store,
    namespace=("team", "shared-evidence"),
)
```

注入的 Store 需要同步 `put/get/search` 和 limit/offset 语义；生命周期由调用方管理。
接口具备扩展点不代表某个持久 provider 已部署。接入时必须再验证检索、元数据、重启、
并发和正确性；数据量变大后还需要原生索引，而非当前的全量枚举排名。

SQLite 原始语料的标签下推、元数据字段投影及核验去重已另做独立消融，见
[STORAGE_OPTIMIZATION.md](STORAGE_OPTIMIZATION.md)。LangGraph 的 `get_metadata`
目前先完整 get 再移除向量，不将接口兼容描述为原生 provider 字段投影。

SQLite 适配器另有 `put_many` 原子批量扩展，支持已有事务中的 savepoint。
外部 Store 没有因此被宣称支持原子批次；其能力需要 provider 单独验证。
接口与对照见 [BATCH_INGESTION.md](BATCH_INGESTION.md)。
