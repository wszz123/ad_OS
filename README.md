# 多智能体低开销协作原型

这是从 LatentMAS 论文复现仓库中隔离出的**社区赛题应用创新版本**，仅包含面向
多 Agent 通信、状态交换、共享记忆、任务评测和恢复验证的代码。
赛题原文见 [`proj_requir.md`](proj_requir.md)；当前版本边界见 [`VERSION.md`](VERSION.md)。

项目重点不是一般工作流编排，而是验证以下系统层机制：

1. 用带 action/params/result/capability/version 的结构化协议降低 Agent 消息开销。
2. 在 Agent 间直接交换 float16 embedding 状态并校验、恢复、使用。
3. 把证据和经验写为共享记忆，供后续 Agent/任务检索复用。
4. 用同条件文本/协议、向量/记忆消融和严格质量检查量化机制作用。

## 架构

```text
Planner ──检索query/state──> Retriever ──facts──> Executor ──tool result──> Summarizer
   │                            │                    │                         │
   └──── Protocol Adapter ──────┴────────────────────┴─────────────────────────┘
                                │
            LangGraph state/checkpoint + State Exchange(SQLite/BLOB)
                                │
                    Memory Adapter(SQLite / LangGraph Store)
                                │
             Evaluator(numeric/unit/citation/traffic/recovery)
```

五层详细说明：

| 模块 | 独立职责 | 核心代码 |
| --- | --- | --- |
| LangGraph | 节点/边、task state、checkpoint恢复 | `coop/cost_graph.py`、`coop/continuous_graph.py` |
| Protocol Adapter | 能力注册/发现、动作和参数/结果校验、消息计数 | `coop/protocol.py`、`coop/text_transport.py` |
| Memory Adapter | 统一数据schema、关键词/标签/向量查询、SQLite/LangGraph Store | `coop/memory.py`、`coop/memory_backends.py`、`coop/sqlite_indexed_memory.py` |
| State Exchange | float32/float16/int8、checksum、SQLite BLOB和恢复 | `coop/state_exchange.py`、`coop/state_store.py` |
| Evaluator | 成本/质量/复用/扫描/故障边界报告 | `tools/run_cost_workload.py`、`coop/answer_validation.py` |

`LangGraph InMemoryStore` 是进程内后端；要跨重启保留记忆，使用 SQLite 后端。
`--durable` 负责状态包和 graph checkpoint 持久化，不会把 InMemoryStore 自动变为持久记忆。

## 快速开始

要求 Linux、Python 3.10。安装赛题原型独立依赖：

```bash
conda create -n coop-prototype python=3.10 -y
conda activate coop-prototype
python -m pip install -r requirements.txt
```

本地完整模型可离线使用：

```text
/home/Users/SharedFile/Models/Qwen/Qwen3-8B
/home/Users/SharedFile/Models/Qwen/Qwen3-Embedding-4B
```

统一入口：

```bash
python -m coop --help
python -m coop doctor --backend qwen
python -m coop demo --backend fixture
python -m coop demo --backend qwen --device cuda:1 --embedding-device cpu
python -m coop summarize --results-dir results --output results/evidence_summary
python -m unittest discover -s tests -v
```

`fixture` 用脚本化 planner 和示意向量检查模块连线，**不是LLM推理或通信性能证据**。
`qwen` 执行四模式真实模型工作流，再分子进程执行状态/纠错恢复。

## 四角色多步任务

使用 Orion 部署成本与 Lyra 归档成本两组各五轮关联任务：

1. Planner 基于公开输入和变量单位目录生成公式/检索请求。
2. Retriever 从实体隔离的多源语料或前序事实记忆获取字段，并核对来源。
3. Executor 执行有界 `Decimal` AST 算术工具，记录每个变量的 source ID。
4. Summarizer 输出金额、CNY单位、引用和解释。
5. Evaluator 检查数字、单位、实际来源和工具输出。金标准不进入 Agent prompt。

公式检查失败时使用事务化 repair ledger 做最多两次修正；预算耗尽明确失败，不计算、不写任务记忆。
CodeAct/任意Python执行沙箱尚未实现；Executor不是CodeAct。

## 四种可复现实验模式

```text
text
protocol
protocol_vector
protocol_vector_memory
```

文本模式是长度标记文本传输加真实当前任务历史，不是由LLM自由解析任意自然语言的MAS基线。
结构化模式与文本模式共用同一模型输入语义和模型预算，以隔离通信编码差异。
向量模式使用Qwen3-Embedding-4B 768维查询向量，状态包float16。
记忆模式保留证据来源并重新计算当前输入，不复用旧答案。

当前报告记录了通信frame token/byte、LLM prompt/output token、向量payload/metadata、
消息数、单任务耗时、事实命中/接受复用、模型调用、语料扫描、纠错尝试、来源及质量标签。
不把扫描JSON字节称为物理磁盘IO，也不从单个小实验推导一般端到端speedup。

## 复现入口

```bash
# 当前四模式真实模型实验
python -m tools.run_cost_workload --device cuda:1 --embedding-device cpu \
  --repair-plan --max-plan-repairs 2 --storage-profile optimized \
  --ingest-strategy batch --durable --repeat 2 \
  --report results/cost_workload.json

# 固定种子新参数、9/137条语料外推及模式级续跑
python -m tools.run_heldout_workload --device cuda:1 --embedding-device cpu \
  --seed 20261010 --distractors 0 --distractors 128 --repeat 2 \
  --report results/heldout.json
python -m tools.run_heldout_workload --resume --report results/heldout.json

# 同条件SQLite参考/索引/投影/去重消融
python -m tools.run_storage_optimization --source-report results/heldout_workload.json \
  --corpus-size 137 --repeat 2 --device cuda:1 --embedding-device cpu \
  --report results/storage.json

# SQLite逐条提交 vs 单事务批量初始化
python -m tools.run_batch_ingestion --source-report results/heldout_workload.json \
  --corpus-size 137 --repeat 2 --workflow --device cuda:1 --embedding-device cpu \
  --report results/batch.json

# 真实Qwen跨进程SIGKILL与纠错ledger恢复
python -m tools.run_recovery_experiment --device cuda:1 --embedding-device cpu
python -m tools.run_repair_recovery --device cuda:1
```

完整部署、指标口径、历史负结果、恢复保证边界见 [`DEPLOYMENT.md`](DEPLOYMENT.md)、
[`WORKLOAD.md`](coop/WORKLOAD.md)、[`STORAGE_OPTIMIZATION.md`](coop/STORAGE_OPTIMIZATION.md)、
[`REPAIR_RECOVERY.md`](coop/REPAIR_RECOVERY.md)。
