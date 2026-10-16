# 未见参数与语料扫描成本验证

## 实验设计

固定种子 `20261010`，生成两组各五个未见参数任务，仍使用 Orion/Lyra 成本场景。
提示、单位/数量合同、纠错上限 2、记忆阈值 0.55 均冻结，运行后不按结果调整。
原始资料价格与费率不变；新参数不复用旧任务的输入组合。

本次参数包括：节点数 7→11、账单天数 43、规划月份 3、保留天数 45、
副本 4→5、流量比例 0.40。保留天数刻意不等于账单天数，以检验错误的变量替代。
不将三个月说成一年，相关问题改称 planning-period。

`coop/heldout_workload.py` 的独立 Decimal 财务评分函数只用于生成标准金额，
不调用运行时 AST 计算器，也不进入 Agent 请求。测试先核对旧手工金额，再用多组
新参数交叉验证独立评分和运行时计算。

两个语料规模：

- 9 条：原始来源和已有两个干扰来源。
- 137 条：保留同样九条，另外加入 128 条其他实体的归档报价，字段名和内容相似。

每种规模做两次反序 trial，对比 `protocol_vector` 与 `protocol_vector_memory`。
每个 mode/trial 都从空任务记忆开始。共计保留 **80 次完整执行**。
实体属于公开请求条件，检索使用实体标签，因此这是参数外推及扫描成本实验，
**不是新领域泛化，也不是同实体内的高难度语义区分实验**。

## 模块

- `heldout_workload.py`：固定种子的参数/语料生成与独立标签。
- `tools/run_heldout_workload.py`：同条件记忆消融、模式级续跑、原子进度保存。
- `MemoryAdapter.scan_stats()`：全量取出和按 ID 点读的次数、行数、向量 JSON 字节数。
- `cost_graph.py`：单独记录语料检索、记忆查询和 online embedding 耗时。
- `prepare_cost_runtime()`：可注入共享模型；支持分批构建向量，避免同时加载第二份 8B。

每个任务保留参数、生成公式、工具数值、引用、纠错轨迹、复用来源和逐任务扫描量。

## 实测结果

报告：`results/heldout_workload.json`。生成模型在 cuda:1，embedding 在 CPU，实测权重bfloat16、输出float32，
八个 CPU 线程，向量 batch size 为 4。两种规模/模式的所有十轮都完成并严格通过：
数值、单位、引用和计算结果均正确。两次 trial 重复结果一致。

| 每十轮（两次均值） | 9 条，无记忆 | 9 条，有记忆 | 137 条，无记忆 | 137 条，有记忆 |
| --- | ---: | ---: | ---: | ---: |
| 严格质量通过 | 10 | 10 | 10 | 10 |
| 模型调用 | 32 | 32 | 32 | 32 |
| 纠错调用 | 12 | 12 | 12 | 12 |
| 接受事实复用 | 0 | 4 | 0 | 4 |
| 语料检索 | 10 | 6 | 10 | 6 |
| 全量扫描取出行数 | 90 | 54 | 1370 | 822 |
| 按 ID 点读行数 | 0 | 57 | 0 | 57 |
| 全量扫描向量 JSON 字节 | 1529860 | 917916 | 23289580 | 13973748 |
| 点读向量 JSON 字节 | 0 | 969090 | 0 | 969090 |
| 两者总向量 JSON 字节 | 1529860 | 1887006 | 23289580 | 14942838 |
| 语料检索耗时，秒 | 0.0133 | 0.0076 | 0.1525 | 0.0940 |
| 记忆查询耗时，秒 | 0 | 0.0070 | 0 | 0.0076 |
| 任务总耗时，秒 | 60.815 | 62.373 | 62.112 | 57.362 |

结论按可验证的作用分开：

1. **新参数复算通过**：更换节点数、副本、保留期、账单期后没有返回旧金额。
2. **实际少扫描**：137 条语料上每十轮少扫描 548 行，即减少 40% 的全量扫描。
3. **计入核验开销**：该规模下原始语料向量 JSON 总加载量净少 8,346,742 字节。
   9 条语料却净增加 357,146 字节，说明小集合下核验点读可能抵消缓存收益。
4. **没有少算模型**：每个 mode 都有 32 次模型调用和相同输入/输出 token。
5. **不推断端到端加速**：检索总计约十分之一秒，而 LLM、CPU embedding、SQLite 写入
   占主要时间。共享主机负载和续跑窗口也会影响总时间，不能把约五秒差异归因于扫描。

扫描字节指标是 SQL 返回后物化的向量 JSON 长度，**不是物理磁盘 I/O，也不包含任务
记忆库或全应用全部数据读写**。计量只用于定位当前语料后端的额外工作。
当前后端在实体过滤前读取全量行；标签条件下推、元数据专用读取和避免重复 get
仍有明显优化空间，后续应独立做消融。

## 构建成本与中断记录

CPU 构建 9 条语料的向量并预热约 14.19 秒；137 条约 293.48 秒，单独记录在 metadata。
该成本没有计入单任务时间，但不能在全生命周期效率结论中省略。

最初运行触及 900 秒工具上限，在大语料第二次 trial 的一个模式中断。
已完成模式及原始工件保留；中断模式从空记忆重跑，未用部分完成的任务冒充完整试验。
续跑复用已构建的语料向量，检查任务定义和 policy/source/vector 指纹一致，不重新构建
137 条向量，也未变更提示、阈值或纠错预算。

报告含 `resume_sessions`、`resume_complete` 和各 mode 工件路径。
历史总墙钟时间在中断前未保存，因此没有人为补算一个总耗时。
这属于**实验模式级续跑**，不等于节点中途恰好一次执行，也不解决跨重启纠错预算续记。
独立的 Planner 预算预留方案已在后续验证，见 [REPAIR_RECOVERY.md](REPAIR_RECOVERY.md)；
未把它回写成此次历史实验模式级续跑的能力。

## 复现与续跑

```bash
conda run -n latentmas python -m unittest discover -s tests -v

OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 conda run --no-capture-output -n latentmas \
  python -m tools.run_heldout_workload --device cuda:1 --embedding-device cpu \
  --seed 20261010 --rounds 1 --distractors 0 --distractors 128 \
  --embedding-batch-size 4 --max-new-tokens 192 --max-plan-repairs 2 \
  --threshold 0.55 --repeat 2 --report results/heldout_workload.json

# 使用保存的配置，跳过完整模式；完成报告只重算摘要，不加载模型
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 conda run --no-capture-output -n latentmas \
  python -m tools.run_heldout_workload --resume --report results/heldout_workload.json
```

`--rounds` 可增加新组合，但仍是已知结构的参数外推。
`--distractors` 可多次指定；数据变大时请保留 embedding 构建成本。
报告按完成模式原子落盘，未完成模式会使用新的工件目录；原目录不会被覆盖。
已有 trial 的语料库、缓存、状态、checkpoint、trace 和 memory snapshot 均保留。

后续已保留参考实现并独立验证标签下推、元数据投影与核验去重，见
[STORAGE_OPTIMIZATION.md](STORAGE_OPTIMIZATION.md)。旧扫描结果没有被覆盖。
