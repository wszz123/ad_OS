# 四角色多来源成本分析：质量与机制分开验证

## 场景及模块

两组各五轮的本地合成任务，均需要跨资料检索、规划公式、执行计算、总结引用。
它们是可复算的工程夹具，不是外部公开 benchmark。

- **Orion 部署**：采购资本成本、月度电费、首月总成本、年度总成本、扩容后年度成本。
- **Lyra 归档**：副本存储费、原始数据出站费、月度综合费用、年度费用、变更副本后的存储费。
- 九条资料：七条适用来源和两条其他实体的干扰来源。价格、功耗、费率和运营成本分散在不同来源。
- 用户指定实体、设备数/天数/副本数等参数；实体标签属于公开任务条件，不是标准答案。

| 模块 | 功能 |
| --- | --- |
| `workload.py` | 资料、公开参数、分场景变量/单位目录；独立评测用金额及引用标准 |
| `cost_graph.py` | Planner → Retriever → Executor → Summarizer，共四类角色 |
| `calculator.py` | Decimal 有界算术；读取 AST，支持变量、括号和 `+ - * /`，不执行 Python 代码 |
| `answer_validation.py` | 最终数值、货币单位、引用 ID、工具结果的独立验收 |
| `plan_validation.py` | 可选计划复核的单位/参数使用检查，不读取标准金额 |
| `plan_repair.py` | 有界纠错、公共数量语义检查反馈、逐次尝试审计及失败终止 |
| `tools/run_cost_workload.py` | 相同任务、提示、预算下的四模式及复核开关评测 |
| `experiment_audit.py` | 通信/质量/复用单独比较，保留错误任务 |

公式由本地 Qwen3-8B 生成，运行时不提供金标准公式或金额。公开变量目录只有含义和单位，
资料数值由 Retriever 获取。独立公式只存在于测试中，用来复核十个手工标注金额。

## 原始金标准

| 任务 | 预期金额（CNY） |
| --- | ---: |
| orion-01 | 21200.00 |
| orion-02 | 144.00 |
| orion-03 | 21644.00 |
| orion-04 | 26528.00 |
| orion-05 | 36992.00 |
| lyra-01 | 288.00 |
| lyra-02 | 64.80 |
| lyra-03 | 352.80 |
| lyra-04 | 4233.60 |
| lyra-05 | 144.00 |

`public_request()` 刻意排除这些金额和必需引用列表。标准答案只用于 graph 返回之后的评分。
工具输出还记录表达式、实际使用的变量、每个数值的来源、实际引用集合。

## 质量口径

一轮通过必须同时满足：

1. 最终金额与标准金额相差不超过 0.005 CNY，拒绝布尔值、NaN/Infinity 等非有限数。
2. 最终单位为 CNY。
3. 引用覆盖标准必需来源，且引用集合等于计算实际使用的来源集合，不能附加编造来源。
4. 工具计算本身的金额及单位也正确，防止总结碰巧报对答案但计算链错误。

这些检查不验证解释文字的全部语义。带正确引用的错误公式仍会失败。
运行状态 `ok` 表示生成流程完成，不等于答案正确；任务失败也计入十轮的质量分母。

## 共享事实缓存

向量模式中，Planner 生成检索向量，State Exchange 保存 float16 字节；Retriever 恢复并用于排序。
复用模式将 Executor 使用过的事实映射、原始来源快照、任务来源和计算结果保存为共享记忆。
缓存向量为来源文档向量的归一化均值。

后续任务只有在实体/配置版本一致、相似度达到预设 0.55、来源未变化、缓存覆盖新公式
所需变量时才能复用。节点数、天数、副本数改变时依然使用新参数重新计算，绝不返回旧金额。
若来源修改或缓存缺字段，回退语料检索。缓存只复用事实，不复用已生成的公式或旧回答。

重新核验快照仍需要解析来源，`source_parses` 包括这部分成本，
`cache_validation_parses` 单独标识它。不能把少了语料检索误报为避免了所有解析。

## 实测及失败记录

报告保留所有阶段，未覆盖失败数据：

- `results/cost_workload.json`：初次提示接口实验，字段和公式都有错误，严格通过率 0/10。
- `results/cost_workload_v2.json`：明确 JSON schema、实体变量目录后的基线，每模式 10/10 流程完成，5/10 质量通过。
- `results/cost_workload_review.json`：增加计划复核，每模式 8/10 流程完成，7/10 质量通过。
- `results/cost_workload_audit.json`：基线及复核的比较与失败任务明细。
- `results/cost_workload_repair.json`：有界纠错，每模式 10/10 完成且 10/10 严格通过，两次 trial。
- `results/cost_workload_repair_audit.json`：基线、单次复核和有界纠错的完整比较。

基线和复核各进行了两次独立 trial，第二次反转模式顺序；任务选择和生成结果可重复。
因GPU0被其他进程使用，本轮为 **Qwen3-8B 在cuda:1，embedding在CPU**；有runtime的报告实测权重bfloat16、向量输出float32，
使用八个 CPU 线程。所有模式在同一设备配置下评测，不能与此前双 GPU 时间直接比较。

### 不启用复核，每 10 轮

| 指标 | 文本 | 协议 | 协议+向量 | 协议+向量+记忆 |
| --- | ---: | ---: | ---: | ---: |
| 流程完成 | 10 | 10 | 10 | 10 |
| 严格质量通过 | 5 | 5 | 5 | 5 |
| 模型调用 | 20 | 20 | 20 | 20 |
| LLM 输入 token | 6608 | 6608 | 6608 | 6608 |
| 通信 token | 35106 | 19157 | 19397 | 19637 |
| 总传输字节，含向量及元数据 | 124857 | 63551 | 81671 | 81961 |
| 语料检索 | 10 | 10 | 10 | 6 |
| 接受事实复用 | 0 | 0 | 0 | 4 |
| 全部来源解析（含缓存核验） | 28 | 28 | 28 | 28 |
| 两次平均任务总耗时，秒 | 26.583 | 26.384 | 43.026 | 43.546 |

- 协议减少通信 token **45.43%**、字节 **49.10%**，模型输入/输出及答案不变。
- 缓存省下四次语料检索，但未省下模型调用或来源解析，当前小数据集上未体现加速。
- CPU embedding 约占向量模式每十轮 16 秒；向量模式不比无向量模式快。
- 相比短句夹具，本轮文本历史更长，协议+向量总字节也低于文本基线；不推断通用压缩率。

### 启用计划复核

`--review-plan` 增加一次 Planner 内部复核，并检查表达式是否忽略公开输入、单位是否一致。
复核不读取标准答案。当前严格质量从 **5/10 提升到 7/10**，但模型调用从 20 增到 28，
两轮被拒绝后没有总结调用，剩余一轮仍是错误公式：

- `orion-03`：复核后仍有不兼容的相加单位，被拒绝。
- `lyra-04`：复核后未使用公开参数 days，被拒绝。
- `lyra-03`：存储副本项语义有误，单位检查不能识别，最终金额验证失败。

这是提升质量并付出调用成本的可选模块，不是已解决所有计算错误。
通过数以全部十个任务为分母，不能以仅八个已完成任务为分母选择性汇报。
这些是单次复核模式的保留结果，新的有界纠错模式如下；运行时不能依据金标准替换公式。

### 有界纠错（最新）

`--repair-plan --max-plan-repairs 2` 使用 `plan_repair.py`，与旧 `--review-plan` 互斥。
草稿生成一次，修改最多两次；JSON 错误和重复无效修改都消耗预算。
第一轮修改也检查看似合法草稿的任务范围，之后仅在确定性检查仍有问题时继续修改。
检查只使用公开问题、变量含义、输入和领域规则，不读取价格数值、标准金额或引用标签。

确定性信号包含 AST/schema、遗漏参数、单位不一致和公共数量语义：
存储按总副本数计算，不额外增加原始副本；流量不计副本；每节点用电按节点数缩放；
显式排除经常性/流量费用时不得包含这些项。数量检查将表达式展开为有界符号项，
能够发现单位检查无法识别的 `1 + replicas` 等偏移项。这些是本成本场景的领域合同，
不是通用数学正确性证明，也没有按 task_id 存储正确公式。

每次尝试记录 `attempt`、`kind`、`candidate`、`issues`、`generation_error`（若有）、
`accepted`。预算耗尽抛出明确异常，保留轨迹，不执行后续计算、不写任务记忆。
所有模型调用/输入输出 token 都计入成本。传输适配器也计入失败前已经发出的请求；
旧报告没有计入失败处理时的请求，因此跨版本通信比较应使用完成状态一致的新实验。
上述原始实验的预算约束单次 Planner 调用。后续已增加 SQLite 原子预留账本，
同一 task 跨进程重放不重置额度，已接受计划零调用重放，详见
[REPAIR_RECOVERY.md](REPAIR_RECOVERY.md)。未知结果仍占额度，不声称外部模型 exactly-once。

两次反序 trial、四模式、每模式十轮，共 **80/80 质量检查通过**。各模式生成结果一致：

| 每十轮 | 基线 | 单次复核 | 有界纠错 |
| --- | ---: | ---: | ---: |
| 流程完成 | 10 | 8 | 10 |
| 严格质量通过 | 5 | 7 | 10 |
| 模型调用 | 20 | 28 | 31 |
| LLM 输入 token | 6608 | 9042 | 11406 |
| LLM 输出 token | 1109 | 1464 | 1641 |

有界纠错每组包含 10 次草稿、11 次修改、10 次总结，预算未耗尽；九轮使用一次修改，
一轮使用两次。相对单次复核多三次调用，补齐此前三轮失败，是质量与成本的明确权衡。
结果只代表本合成语料，解释文字和其他领域的泛化仍未验证。

在有界纠错模式中：

- 文本和协议的模型 token、数值、引用完全一致；通信 token 为 35178 与 19201，减少 **45.42%**。
- 文本/协议总传输字节为 125210/63730，减少 **49.10%**。
- 向量+记忆模式接受四次事实复用，语料检索 10→6；仍需要重新规划/计算，不省模型调用。
- 来源解析仍计入缓存核验；CPU embedding 与持久写入的耗时未排除。
- 两次平均任务时间为文本 37.931 秒、协议 37.597 秒、向量 53.326 秒、向量+记忆 54.121 秒；没有整体加速证据。

## 复现

```bash
conda run -n latentmas python -m unittest discover -s tests -v

OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 conda run --no-capture-output -n latentmas \
  python -m tools.run_cost_workload --device cuda:1 --embedding-device cpu \
  --max-new-tokens 192 --threshold 0.55 --repeat 2 --durable \
  --report results/cost_workload_v2.json

OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 conda run --no-capture-output -n latentmas \
  python -m tools.run_cost_workload --device cuda:1 --embedding-device cpu \
  --max-new-tokens 192 --threshold 0.55 --review-plan --repeat 2 --durable \
  --report results/cost_workload_review.json

conda run -n latentmas python -m tools.audit_cost_workload \
  --report results/cost_workload_v2.json --report results/cost_workload_review.json \
  --output results/cost_workload_audit.json

OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 conda run --no-capture-output -n latentmas \
  python -m tools.run_cost_workload --device cuda:1 --embedding-device cpu \
  --max-new-tokens 192 --threshold 0.55 --repair-plan --max-plan-repairs 2 \
  --repeat 2 --durable --report results/cost_workload_repair.json

conda run -n latentmas python -m tools.audit_cost_workload \
  --report results/cost_workload_v2.json --report results/cost_workload_review.json \
  --report results/cost_workload_repair.json --output results/cost_workload_repair_audit.json
```

两张 GPU 都空闲时，可改用 `--device cuda:0 --embedding-device cuda:1`；应生成新报告。
工件目录保留逐任务协议帧、SQLite 语料、记忆、状态、checkpoint 和公式结果。

## 后端扩展

`--memory-backend sqlite|langgraph` 可以替换任务记忆，Agent 图不变。
运行器通过统一 `items/count` 等接口导出记忆，不再依赖后端 SQL 连接。
两后端采用相同排名规则；选择原生索引会是另一项消融。
完整对照见 [MEMORY_BACKENDS.md](MEMORY_BACKENDS.md)。

`tools/run_heldout_workload.py` 冻结策略后生成新参数，比较9/137条语料的记忆消融。
所有保留的80次执行通过；总加载量统计包括核验点读，不只报告减少的全扫描。
结果与模式级续跑说明见 [HELDOUT.md](HELDOUT.md)。
