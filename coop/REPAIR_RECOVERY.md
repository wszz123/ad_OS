# Planner 节点中途重跑：纠错预算持久化

## 与图 checkpoint 的边界

LangGraph 的 checkpoint 位于节点边界。原先 Planner 的多次 LLM 调用在同一个节点内，
进程在节点返回前被杀，重跑该节点会重新获得完整纠错次数。
本轮不试图让 LangGraph 在函数中途 checkpoint，而是增加独立的持久调用额度账本。

- `checkpoints.sqlite`：图输入、当前节点、节点完成后的最终计划/指标。
- `repair_ledger.sqlite`：节点内部每次模型调用的额度预留、结果及策略指纹。

两个数据库不是一个分布式事务。ledger 在调用前提交，checkpoint 可以仍停留在
`next=("planner",)`；新进程重跑 Planner 时先查询 ledger。

## 接口与语义

`coop/repair_ledger.py` 使用 WAL、synchronous=FULL，保存：

1. task 的策略 manifest：fingerprint、max_repairs。
2. 每个 attempt 的编号、剩余修复次数、started/rejected/accepted、候选/检查信息及预留 token。

`BEGIN IMMEDIATE` 原子分配下一个编号，调用开始之前提交额度预留。
默认最多一次草稿（attempt 0）和两次修复（attempt 1、2）；修复次数依次为2、1、0。
完成后只能更新自己尚未完成的预留记录，不能用 INSERT OR REPLACE 改写历史或扩展配额。

恢复规则：

- `started` 且结果未保存：视为已用额度，不能重新执行同一 attempt。
- 已接受计划：重新校验后直接返回，零模型调用。
- 三个额度已经用完：明确 PlanRepairError，零新增模型调用。
- 有先前已完成候选：供下一个修复参考；未知结果保留为未知，不伪造候选。
- prompt、公开输入、问题、catalog、生成预算、纠错上限、checker version 或 policy context
  改变：拒绝使用原账本，不静默重置预算。

`cost_graph.py` 将语料/模型/策略 namespace 作为 policy context。
正常 `--repair-plan` 运行自动在 mode 工件中创建账本，最终尝试历史仍参与 graph state/报告。
账本的持久化元数据不进入 LLM 的历史提示，因此未中断时的提示和生成行为保持一致。

这是**跨进程至多预算预留**，不是外部 LLM exactly-once。
进程可以在预留后、实际调用前死亡，损失一个尚未用到的额度；也可以生成完成但结果未知。
保守计费可以保证上限，不能恢复丢失的外部结果或推断未知 token 消耗。

## 可核查的 SIGKILL 实验

`tools/repair_recovery_worker.py` 运行一个真正的 LangGraph Planner 节点。
控制器 `tools/run_repair_recovery.py` 只杀死自己创建的 prepare worker，恢复 worker 重开
相同 ledger/checkpoint，以相同 thread 的 `invoke(None)` 重跑 Planner。

真实 Qwen3-8B 的两个场景：

| 场景 | 杀死位置 | 恢复后的模型调用 | 结果 |
| --- | --- | ---: | --- |
| inflight | 草稿已生成，ledger 结果尚未提交 | 1 | 未重复草稿，新增一次修复后接受 |
| accepted | ledger 已接受，Planner 节点尚未 checkpoint | 0 | 返回原计划，完成图 |

两种情况都观察到恢复前 checkpoint 的下一节点是 Planner，原预留记录完整保留，
编号连续、剩余预算正确、物理调用和预留数都不超过3。
完成后的图 reinvoke，以及直接重调同一纠错函数，均没有额外生成。

真实报告：`results/repair_recovery_v2.json`，全部检查通过。
fake 模型另验证耗尽场景：三次拒绝后恢复零调用失败，报告为
`results/repair_recovery_fake.json`。fake 测试也实际使用子进程和 SIGKILL。
早期 `results/repair_recovery.json` 仅保留为开发记录，其简单检查不足以证明正确的
预留/已接受重放语义，正式结论采用 v2 报告。

普通四角色工作流也重新验证：`results/cost_repair_ledger.json` 中十轮全部完成并严格通过，
31次模型调用、11406输入 token、1641输出 token，与此前未中断实验一致。

## 复现

```bash
conda run -n latentmas python -m unittest discover -s tests -v

# 快速：真实进程、fake 响应，包含耗尽场景
conda run -n latentmas python -m tools.run_repair_recovery --fake \
  --report results/repair_recovery_fake.json

# 真正加载本地 Qwen，只验证非篡改模型输出的两个场景
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 conda run --no-capture-output -n latentmas \
  python -m tools.run_repair_recovery --device cuda:1 --max-new-tokens 192 \
  --report results/repair_recovery_v2.json

# 四角色正常工作流，保存 ledger、状态和 graph checkpoint
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 conda run --no-capture-output -n latentmas \
  python -m tools.run_cost_workload --device cuda:1 --embedding-device cpu \
  --mode protocol_vector_memory --repair-plan --max-plan-repairs 2 \
  --storage-profile optimized --ingest-strategy batch --durable \
  --report results/cost_repair_ledger.json
```

## 验收与运维范围

64项测试通过，包含数据库重开、重复进程死亡不增加额度、已接受零调用重放、
耗尽零调用失败、策略变更拒绝、两个连接原子预留，以及三种 SIGKILL 场景。

额度分配在多连接下原子，但还没有任务租约/调度所有权系统；正常使用仍应保证每个
task 只有一个活动 worker。并发冲突会拒绝分派或保守消耗额度，不承诺并发合并结果。
已完成/耗尽账本应按保留策略归档；此原型随实验保留，不自动删除。
无 session manifest 的早期 ledger 拒绝恢复，不尝试猜测已用预算。
