# 录制脚本：低开销多Agent协作原型

建议时长4–5分钟。正式录制使用`qwen`结果；fixture只用于预演，画面中必须标明。

## 0:00–0:35：赛题目标与边界

展示`proj_requir.md`及`coop/ARCHITECTURE.md`。
说明系统要比较文本/协议消息、传递embedding、复用跨任务记忆并实际计量。
指出LangGraph负责运行时，不把普通编排当成主要机制贡献。

## 0:35–1:15：五层模块

展示 `coop/` 文件结构：

1. LangGraph运行时。
2. Protocol Adapter：统一action/params/result/capability。
3. Memory Adapter：SQLite/LangGraph Store同接口。
4. State Exchange：查询向量float16二进制传递。
5. Evaluator：按消息、token、byte、调用、质量和耗时拆分报告。

强调当前实现交的是embedding状态，不是将原论文latent KV结果冒充为已交付功能。

## 1:15–2:20：四角色任务

打开一个heldout任务报告，按trace说明Planner→Retriever→Executor→Summarizer：
公开设备/月份参数被转换为公式；Retriever返回带来源事实；Executor执行有界Decimal AST；
Summarizer返回value/unit/citation；Evaluator在Agent之外校验金额和来源。

展示plan attempt ledger：最多2次修复，超预算失败，不回退使用金标准公式。

## 2:20–3:10：通信和共享记忆

在同一任务模式对照text frame与compact JSON frame，指出两者LLM prompt token相同，
消息token减少是编码开销的变化。展示状态包state ID/dtype/shape/checksum，和后续线程复用
事实但按新参数重新计算。

报告示例：四模式有界纠错每十任务严格通过10/10；协议相对text的通信token减少约45.42%。
向量+记忆接受4次事实复用、语料检索10→6，但没有减少模型调用；不宣称整体加速。

## 3:10–4:10：存储优化和恢复

- 137条corpus：标签下推、元数据投影、核验去重将逻辑向量JSON物化约14.94MB降到0.85MB；
  这是物化数据，不是物理磁盘IO，端到端时间并未改善。
- 语料初始化batch提交137→1，写入约20.14秒→0.246秒；事务和同步级别保持。
- 两个恢复演示：Planner checkpoint后状态恢复；节点内部ledger预留保证预算不重置。
  说明未知LLM结果仍计额度，不承诺外部调用exactly-once。

## 4:10–4:45：运行与交付

展示`python -m coop demo --backend qwen ...`，打开新生成目录的manifest、trace、SQLite和summary。
展示独立源码ZIP及SHA256清单。结束时说明合成场景范围、未完成项和正式视频/提交状态。

## 录制命令

```bash
conda run -n coop-prototype python -m coop doctor --backend qwen
conda run --no-capture-output -n coop-prototype python -m coop demo --backend qwen \
  --device cuda:0 --embedding-device cuda:1 --cpu-threads 8
```

真实演示单轮耗时较长；推荐后台先跑并保留输出，再用实际生成的同一报告录屏讲解。
不得将fixture输出称为真实Qwen结果，也不要剪掉失败、耗时口径或未实现功能说明。
