# 原子批量写入与初始化成本

## 目的

此前137条语料的预加载逐条提交，耗时约十几至二十秒。
本轮只更换初始化的提交粒度：保留原 `per_item`，增加 `batch`，不改日志/同步设置，
不批量延迟保存任务执行期间的新记忆。

## SQLite 接口

`MemoryAdapter.put_many(items)` 返回输入顺序的记忆 ID 列表。
`IndexedSQLiteMemoryAdapter` 继承该接口，标签触发器与源记录在同一事务中更新。

- 空批次返回空列表，不创建写事务。
- 重复 ID 与逐条 upsert 一致，最后一次写入生效。
- 编码/embedding 准备在获得写锁前完成；准备失败不会写入部分记录。
- 没有调用方事务时，一次 BEGIN、整批 INSERT/REPLACE、一次提交。
- 已有事务时使用嵌套 savepoint，不提交或回滚调用方的其他工作。
- 中途数据库约束失败，批次的插入、已有记录更新、派生标签更新一起回滚。
- `put(item)` 复用相同逻辑，一条记录仍对应一次独立提交；如果处在调用方事务中，
  只释放自己的 savepoint，提交权由调用方持有。

这是一项 SQLite 扩展能力，未将原子批量写入要求强加给 LangGraph BaseStore。
数据库回滚也不回滚调用方 Python 对象的 ID/时间/向量填充，或外部 embedding 回调副作用。
批量数据先物化为行参数，超大数据集应由调用方选择分块粒度；分块不等于全数据集一个事务。

`write_stats()` 记录适配器 API 的事务、savepoint、提交、回滚与接受行操作数，
不包含 schema/index 初始化或调用方原始 SQL。`rows_accepted_in_scopes` 是接受的写操作数，
不是去重后记录数；savepoint 释放后，调用方仍可回滚其外层事务。

## 实验

`tools/run_batch_ingestion.py` 复用已保存的137条768维向量资料，使用带标签索引的 SQLite。
两次试验，第二次反转执行顺序；在新文件上比较逐条提交和单事务提交。
计时包含行准备、JSON 序列化、SQL 写入和提交，schema/index 构造及重开验证单独在计时外。

报告：`results/batch_ingestion.json`。

| 写入137条记录，两次平均 | 逐条提交 | 批量事务 |
| --- | ---: | ---: |
| 写入时间 | 20.1389秒 | 0.2462秒 |
| API 提交次数 | 137 | 1 |
| journal_mode | delete | delete |
| synchronous | FULL（2） | FULL（2） |
| 重开后的记录/检索 | 一致 | 一致 |

内容摘要、输入 ID 顺序、记录数量、重开后查询与提交次数全部通过验证。
没有关闭日志、禁用 fsync 或降低 synchronous 来换取该结果。
这是当前文件系统和小批次上的初始化收益，不推断通用存储吞吐。

另运行两次真实 Qwen 配对工作流，共 **40/40** 次任务通过金额/单位/引用检查。
公式、答案、纠错轨迹、模型 token、通信量和4次事实复用保持一致。
这些验证固定 `optimized` 存储查询配置，仅切换 corpus 初始化方式。
工作流内部新记忆仍逐轮提交，未把新任务结果放在一个长事务里延迟可见。

## 复现

```bash
conda run -n latentmas python -m unittest discover -s tests -v

# 只测写入与重开，不加载模型
conda run -n latentmas python -m tools.run_batch_ingestion \
  --source-report results/heldout_workload.json --corpus-size 137 --repeat 2 \
  --report results/batch_ingestion_only.json

# 同时验证真实模型工作流
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 conda run --no-capture-output -n latentmas \
  python -m tools.run_batch_ingestion --source-report results/heldout_workload.json \
  --corpus-size 137 --repeat 2 --workflow --device cuda:1 --embedding-device cpu \
  --report results/batch_ingestion.json

# 常规运行可组合查询优化与批量初始化
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 conda run --no-capture-output -n latentmas \
  python -m tools.run_cost_workload --storage-profile optimized --ingest-strategy batch \
  --mode protocol_vector_memory --repair-plan --durable \
  --device cuda:1 --embedding-device cpu --report results/cost_optimized_batch.json
```

每种写入策略的数据库、工作流的状态/checkpoint/trace/记忆快照均保留。
报告独立保存 source 配置、SQLite 版本、durability 设置与 API 写入统计。
59项测试通过，涵盖失败回滚、savepoint 隔离、重复 ID、索引同步、重开一致性及工作流等价。
