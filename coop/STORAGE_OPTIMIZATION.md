# SQLite 标签下推、元数据投影与缓存核验去重

## 目标及对照边界

上一轮发现：原始语料先取出全表再过滤；核验缓存来源时同一记录重复 get，
每次还载入不需要的768维向量。本轮逐项优化数据访问，保持 Agent、任务、语料、
精确排序、模型、提示、纠错和复用阈值不变。

保留四个显式配置，默认仍是参考实现，便于复现旧实验：

| `--storage-profile` | 标签下推 | 元数据投影 | 去重核验 |
| --- | --- | --- | --- |
| `reference` | 否 | 否 | 否 |
| `tag_pushdown` | 是 | 否 | 否 |
| `metadata_projection` | 是 | 是 | 否 |
| `optimized` | 是 | 是 | 是 |

本轮优化原始资料库，不改变任务记忆后端；成本图仍使用同一 MemoryBackend 接口。

## 实现

### 标签下推

`coop/sqlite_indexed_memory.py` 中的 `IndexedSQLiteMemoryAdapter` 继承参考适配器。
源表 `memories` 的格式不变；新增派生表 `coop_memory_tags`，主键为 `(tag, memory_id)`。
SQL 先找满足所有标签的 ID，再关联并取出源记录，之后仍调用相同的精确关键词/余弦排序。
重复标签按集合处理，并列分数按 memory_id 排序；所有标签值都使用绑定参数。
无标签查询保持全量语义，不将标签索引称为 ANN 或向量数据库。

SQLite INSERT/UPDATE/DELETE 触发器同步派生标签，包括旧 `MemoryAdapter` 连接上的写入。
构造索引时从 tags_json 回填，重建派生数据不读取 embedding 字段；原记录不被删除。
单独记录 `index_setup_sec` 及语料逐条写入耗时，不把索引视为免费。

### 元数据投影

统一接口新增 `get_metadata(id)`，返回同一 `MemoryItem` 元数据但 embedding 为 None。
SQLite 查询显式选择元数据列，使用 NULL 代替 embedding_json；不会读取并反序列化向量。
只有来源 ID/摘要/evidence 的核验改用该 API，需要向量的汇总仍使用完整 get。
测试用损坏的 embedding JSON 验证投影不会偷偷解码该字段。

LangGraph BaseStore 没有统一字段投影 API，其适配器只能先取完整值再移除向量。
因此该后端的接口兼容并不代表 provider I/O 下降；本轮投影收益仅验证了 SQLite。

### 核验去重

参考路径保留同一来源两次 get，作为独立消融。
优化路径一次读取保存在局部变量中，再检查存在性和来源快照。
来源修改仍会拒绝缓存并回退检索；没有放松事实完整性或引用检查。

## 实际结果

使用上一轮已保存的137条语料向量和10个未见参数任务，不重新构建 embedding。
Qwen3-8B 在 cuda:1，query embedding 在 CPU，8个线程；实测模型权重bfloat16、向量输出float32，最多两次纠错。
两次 trial，第二次反转配置顺序，各配置从空任务记忆开始。
报告：`results/storage_optimization.json`。

全部 **80/80** 次执行完成且通过金额/单位/引用/工具检查。
任务公式、答案、纠错记录、复用 ID/来源、模型 token 和通信量与参考配置一致。
每配置仍为32次模型调用、4次接受复用、6次语料检索。

| 每十轮，两次均值 | 参考 | 标签下推 | +元数据投影 | +核验去重 |
| --- | ---: | ---: | ---: | ---: |
| 语料候选记录取出行数 | 822 | 21 | 21 | 21 |
| 完整来源 get | 57 | 57 | 29 | 29 |
| 元数据 get | 0 | 0 | 28 | 14 |
| 总向量 JSON 加载字节 | 14942838 | 1326147 | 850071 | 850071 |
| 语料检索耗时，秒 | 0.0910 | 0.0036 | 0.0040 | 0.0040 |
| 索引构造耗时，秒 | 0 | 0.580 | 0.642 | 0.475 |
| 新语料逐条写入耗时，秒 | 16.777 | 19.927 | 18.343 | 14.718 |
| 任务总耗时，秒 | 55.081 | 58.397 | 58.776 | 58.662 |

已验证的作用：

- 标签下推减少822→21条候选记录物化，避免将128条其他实体资料的向量载入 Python。
- 元数据投影额外省去476076字节来源向量字段反序列化；汇总仍须完整读取29条记录。
- 去重使来源核验的元数据读减半，28→14；它不进一步降低向量字节，因为投影已排除了这些向量。
- 完整组合使总向量 JSON 加载量约减少 **94.31%**。

该字节数包括全量候选、过滤候选和完整 get 的向量字段，**不是物理磁盘 I/O**，
不包含元数据自身或任务记忆库的全部流量。
局部检索耗时下降，但不足以证明全流程加速：本轮总任务时间未改善，模型、CPU
embedding、持久化及共享主机波动占主要时间。索引及写入耗时也必须保留。

## 运行

使用已有 `results/heldout_workload.json` 及其工件目录；若文件不在，请先按 HELDOUT.md 运行。

```bash
conda run -n latentmas python -m unittest discover -s tests -v

OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 conda run --no-capture-output -n latentmas \
  python -m tools.run_storage_optimization \
  --source-report results/heldout_workload.json --corpus-size 137 --repeat 2 \
  --device cuda:1 --embedding-device cpu --report results/storage_optimization.json

# 常规成本任务可显式选择优化组合
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 conda run --no-capture-output -n latentmas \
  python -m tools.run_cost_workload --storage-profile optimized \
  --mode protocol_vector_memory --repair-plan --durable \
  --device cuda:1 --embedding-device cpu --report results/cost_optimized.json
```

对照器校验当前任务清单及源/模型/策略指纹，每个完成配置原子保存结果。
各配置保留 corpus、task memory、二进制状态、checkpoint、trace 与 memory snapshot。
`filtered_scan_rows` 与 `full_scan_rows` 分开计量；比较总加载量时必须包括
`filtered_vector_json_bytes_loaded`，不能把零全表扫描误报为零数据读取。

## 当前验收与下一步

54项测试通过，包含多标签、并列排序、特殊字符绑定、旧连接写入索引同步、损坏向量
下的元数据投影、源修改缓存失效、四配置答案一致及实际数据减少。
下一步可独立验证批量写入、二进制向量存储或原生索引，以及跨重启的纠错预算续记。

批量写入现已单独验证：在相同 durability 设置下，137条语料的 API 提交137→1，
平均初始化20.14秒→0.246秒，检索与真实任务结果一致。见 [BATCH_INGESTION.md](BATCH_INGESTION.md)。
