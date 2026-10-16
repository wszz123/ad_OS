# 架构与模块边界

## 总览

```text
                    LangGraph StateGraph
              state/checkpoint/control-flow recovery
                            │
       ┌────────────────────┼─────────────────────┐
       │                    │                     │
 Planner ──protocol──> Retriever ──protocol──> Executor ──protocol──> Summarizer
       │                    │                     │                     │
       ├── State Exchange   ├── Memory Adapter    └── bounded Decimal tool
       │                    │
       └──── embedding       └── SQLite / LangGraph Store
                            │
                    Evaluator / audit artifacts
```

系统按五个Adapter边界组合。LangGraph负责运行时和状态流转；协议、记忆、向量状态、
工具执行和评测均由独立模块实现，便于替换并逐项消融。

## 模块构成

### 1. LangGraph：`coop/cost_graph.py`

四个角色节点沿Planner→Retriever→Executor→Summarizer执行。TaskState传递task ID、
用户参数、公式、state ID、事实和结果引用。持久checkpointer按thread保存节点边界状态。

### 2. Protocol Adapter：`coop/protocol.py`、`coop/text_transport.py`

`Capability`声明agent ID、role、action和protocol version。`Message`携带source、target、
action、params、message ID和version；返回结果使用同一封套关联`in_reply_to`。
注册时检查能力/handler一致性；运行时检查version、端点、action和JSON可序列化结果。

提供文本和JSON两种wire transport。文本为长度标记字段并携带本任务实际历史；协议为
紧凑JSON。各Agent调用相同模型提示，Evaluator分别统计wire消息token与LLM prompt token。

### 3. Memory Adapter：`coop/memory.py`、`coop/memory_backends.py`

统一`MemoryItem` schema：记忆ID、来源Agent、UTC创建时间、主题、摘要、类型、tags、
evidence/provenance和可选embedding。统一接口为put/get/get_metadata/items/count/search/
search_vector/close。SQLite为持久本地后端；LangGraph InMemoryStore为同进程后端。

SQLite查询排序使用可替换profile：全表参考实现、tag下推、元数据投影、来源读取去重。
标签索引由触发器同步。`put_many`用一事务预加载语料，任务产生的记忆仍逐项提交。

### 4. State Exchange：`coop/embedding.py`、`coop/state_exchange.py`、`coop/state_store.py`

Qwen3-Embedding-4B使用last-token pooling、归一化、768维截断产生float32查询向量，
再编码为float16。Packet含state ID、源/目标Agent、dtype、shape、encoding和SHA256。
SQLiteStateStore把metadata和binary BLOB分开持久化；Retriever恢复后直接参与余弦排序。

这证明的是embedding状态的生成/保存/恢复/使用，不是LatentMAS隐藏状态或KV-cache移交，
也不是Socket/共享内存transport。

### 5. Executor与Evaluator：`coop/calculator.py`、`coop/answer_validation.py`

Executor只解析受限算术AST（Decimal、数值、命名变量、括号、加减乘除），不调用eval/exec。
Evaluator以agent外部标签检查数值、CNY单位、来源集合和工具计算结果；金标准不进prompt。
该工具不是CodeAct，也不是Python安全沙箱。

## 信息和持久化边界

| 信息 | 存放位置 | 生命周期 |
| --- | --- | --- |
| 当前task、下一节点、结果引用、过程指标 | LangGraph state/checkpoint | 单thread，可恢复节点边界 |
| 原始事实及文档embedding | corpus SQLite | 持久 |
| 跨task经验、摘要、证据链 | memory SQLite或注入Store | 依provider；InMemoryStore只在当前进程 |
| 查询向量二进制 | state SQLite BLOB | 持久、checksum校验 |
| 纠错调用预留与attempt结果 | RepairLedger SQLite | 跨进程保留调用预算 |
| 完整frame、模型统计、逐任务答案 | artifact JSON/trace | 实验审计 |

RepairLedger在LLM调用前以`BEGIN IMMEDIATE`预留额度，防止Planner节点重放重置最大次数。
这限制调用次数，不提供外部模型exactly-once：若调用结果未知，已预留额度仍作废。

## 可复现实验构成

- 原任务集：Orion与Lyra各5个关联task。
- 未见参数集：seed=20261010生成变化的节点数、天数、月份、保留期、副本、流量比例。
- 语料规模：9条原始/干扰source；另加128条异实体归档干扰source，共137条。
- 通信模式：text、protocol、protocol+vector、protocol+vector+memory。
- 计划质量策略：无修复、单次review、有界repair；最多两次repair。
- SQLite storage策略：reference、tag_pushdown、metadata_projection、optimized。
- ingestion策略：per_item、atomic batch。
- 记忆provider：SQLite、LangGraph InMemoryStore。
- 故障实验：Planner后checkpoint恢复、节点内Ledger预算SIGKILL恢复。

任务顺序、prompt、模型、greedy解码预算和已声明的阈值都在报告内记录。
双trial反转模式顺序；失败保留并进入分母。fixture与真实Qwen证据分别标记。
不将消息payload字节当物理I/O，不将局部检索耗时下降写成端到端speedup。

## 文件地图

```text
coop/        五层adapter、四角色图、数据模型与版本化协议
tools/       workload、后端/存储消融、恢复控制器、审计及CLI
tests/       协议/语料/公式/引用/事务/索引/重放恢复测试
results/     运行生成，gitignore排除
```

```bash
python -m coop doctor --backend qwen
python -m coop demo --backend qwen --device cuda:0 --embedding-device cuda:1
python -m unittest discover -s tests -v
```
