# 持久状态传递与跨进程恢复

## 可验证的小方案

本轮验证的是：**Planner 已执行完成并同步提交 checkpoint 后，进程被 SIGKILL，
新进程无需再次规划/生成向量即可完成检索和总结。**

拆分为四个持久文件，保持五层架构边界：

| 文件 | 所属层 | 内容 |
| --- | --- | --- |
| `corpus.sqlite` | Memory Adapter | 原始知识语料及文档向量 |
| `memory.sqlite` | Memory Adapter | 完成任务后形成的结论、证据及来源 |
| `states.sqlite` | State Exchange | 带 ID、来源/目标 Agent、dtype/shape/checksum 的二进制向量 |
| `checkpoints.sqlite` | LangGraph | thread 状态、待执行节点、过程指标与状态引用 |

图状态只保存 `state_id`；Retriever 从状态库恢复 float16 向量，用于语义检索。
原进程内对象已随 SIGKILL 消失，新进程使用的是数据库中的向量。
这个后端是基于本机持久文件的进程间状态交接，不是 Socket 或共享内存传输。

## 运行

在仓库根目录、Python 3.10 环境中：

```bash
conda run -n latentmas python -m pip install -r coop/requirements.txt \
  --proxy http://127.0.0.1:1081

# 快速：fake 模型，实际两个进程与 SIGKILL
conda run -n latentmas python -m tools.run_recovery_experiment --fake \
  --report results/fake_recovery.json

# 实际加载本地 8B 与 embedding 模型
conda run --no-capture-output -n latentmas python -m tools.run_recovery_experiment \
  --device cuda:0 --embedding-device cuda:1 --max-new-tokens 96 \
  --report results/qwen_recovery.json

# 持久化后端的十轮连续执行
conda run --no-capture-output -n latentmas python -m tools.run_qwen_continuous \
  --mode protocol_vector_memory --durable \
  --report results/qwen_continuous_durable.json
```

恢复控制器只终止自己创建的 worker；每次运行创建独立工件目录。
`prepare.json`、`resume.json` 保存进程 ID、调用角色、token、状态 checksum、
向量字节数、指标、协议帧及回答。`prepare.log`、`resume.log` 保存模型日志。

## 验收结果

真实本地模型实验的 11 项检查全部通过：

1. 两个 worker 的 PID 不同，首进程退出码为 SIGKILL。
2. 首 checkpoint 的下一节点为 Retriever。
3. 恢复前后的 1,536 字节向量 checksum 一致，状态记录仍为 1 条。
4. 恢复进程只生成总结，未调用 Planner、未重新加载 embedding 模型或编码 query。
5. 检索命中原证据、答案生成完成。
6. 任务记忆从 0 条变为 1 条。
7. 完成后再次 `invoke(None)` 不新增生成调用。
8. 前序 embedding 耗时等指标从 checkpoint 恢复。

完整报告：`results/qwen_recovery.json`。
首进程生成计划使用 49 输入/28 输出 token；恢复进程总结使用 72 输入/33 输出 token。
恢复必须重新加载生成模型，因此本轮不将模型加载耗时省略为恢复加速指标。

## 后端约束与恢复范围

- SQLite 状态行以 `(namespace, state_id)` 为键；namespace 用于隔离不同模型/配置。
- 状态先持久提交，再保存引用它的 planner checkpoint。
- 同 ID、同内容写入可重试；不同内容写入拒绝，防止旧 checkpoint 指向被改写的向量。
- 恢复读取检查 checksum、dtype、shape、载荷长度和有限浮点值。
- Graph 的 `task_metrics` 参与 checkpoint，恢复节点不依赖旧进程的 Python 指标字典。
- 恢复使用同一 graph/config、同一 `thread_id` 的 `invoke(None)`，无需重新提交任务输入。

四个数据库之间没有分布式事务。节点中途退出可能留下未被 checkpoint 引用的状态，
也可能导致节点重跑；本轮不证明任意故障位置的 exactly-once 语义。
任务 ID 在一个 namespace 内必须唯一，状态不可变；连续实验通过独立运行目录隔离各轮。
后续应验证节点内故障、孤立状态清理、配置迁移与并发任务，之后再比较 Socket/共享内存后端。

Planner 内部的跨进程预算现由独立预留账本验证，见 [REPAIR_RECOVERY.md](REPAIR_RECOVERY.md)。
它不改变上述状态/checkpoint 边界：新进程重跑 Planner，但从 ledger 读取已用额度，
不会重新获得完整预算。真实草稿结果未提交及已接受未 checkpoint 两种场景均通过。
