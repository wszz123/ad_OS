# V2：多文件、有状态的目录服务任务

## 和 V1 的实质区别

V1 是一个单文件搜索函数的五类参数处理缺陷。V2 是一个实际可执行的多模块目录服务练习：

- 8 个代码模块、3 份有版本含义的文档。
- SQLite 保存三个租户的 360 条初始商品记录、数据和策略版本；B3 前新增一条记录。
- 候选/部署版本分别管理，每轮修复累计保留。
- 每个复现场景包含多次读、写、配置变更或重试，并保留该场景内的缓存、revision、receipts。
- 单次修复通常需要改 2–3 个文件；只修复一个表象不能通过检查。
- 完整链路仍为 `B1 → A1 → B2 → A2 → … → B5 → A5`，两组各五轮。

这是**自建项目任务集**，不是来自真实生产日志或第三方公开 benchmark。场景执行的是实际
Python 模块和 SQLite 读写，不代表已经部署了常驻 HTTP 服务、真实并发流量或大型软件仓库。

## 模块结构

```text
service.py
  ├─ repository.py    租户数据选择
  ├─ policies.py      全局/租户策略合并与类型解释
  ├─ limits.py        数量边界与租户上限
  ├─ cache.py         请求、版本及策略相关的缓存键
  └─ pagination.py    稳定排序、游标生成与校验

gateway.py            写入入口、幂等回执与乐观版本检查
  └─ mutations.py    批量库存更新、提交与回滚语义

docs/architecture.md  当前架构与跨模块契约
docs/policy.md        当前有效业务规则
docs/legacy-v0.md     明确标记失效的历史规则
```

代码实现及原始缺陷见 `coop/catalog_project.py`，环境和验收见 `coop/catalog_world.py`，
受限子进程执行器见 `coop/catalog_worker.py`。参考修复配方只供 fixture 决策器使用，
不会加入 Qwen 的任务、模型可见项目文件 bundle 或工具观察上下文。

## 两组连续任务

| 运维任务 | 独立交付 | 研发任务 | 独立交付与涉及文件 |
| --- | --- | --- | --- |
| B1 | 在租户交替请求中复现数据/缓存串租，收集当前版本与日志 | A1 | 同时修复 repository 选择与 cache 租户命名空间，2 个文件 |
| B2 | 部署 A1，检查租户契约，再复现重复名称分页漏项及游标串用 | A2 | 稳定复合排序、游标绑定及分页缓存键，pagination/cache，2 个文件 |
| B3 | 数据版本变化后部署 A2，复现写后读取旧库存、旧游标未失效 | A3 | 写入 revision 与缓存版本协同修复，mutations/cache，2 个文件 |
| B4 | 策略版本变化后部署 A3，复现显式 false、动态策略缓存与租户上限错误 | A4 | policies/limits/cache 协同修复，3 个文件 |
| B5 | 部署 A4，复现部分更新、重试重复扣减、幂等键串租及过期写入 | A5 | mutations/gateway 实现原子性、回执及乐观并发检查，2 个文件 |

运维完成意味着**复现当前故障并确认前序契约**，不是故障已修复；研发完成意味着候选版本
通过本轮和前序回归，之后才由下一轮运维部署。没有前置成功，后续依赖任务标记 blocked。
不会注入参考补丁或把未执行的任务记作成功。

## 例子一：为什么 A1 不能只改一个文件

三个租户可有相同的商品 ID 和商品名，但库存不同。

```text
读取 red / shared  → 应只得到 red 的数据
读取 blue / shared → 应只得到 blue 的数据
再次读取 red      → 应可缓存命中且仍属于 red
```

初始 repository 不过滤租户；初始 cache 只按 query 建键。
即使修好 repository，第二次请求仍可能拿到第一个租户的缓存。因此验收分别检查租户记录
合法性、缓存命名空间、重复请求一致性和真实缓存命中；直接禁用缓存也不能蒙混通过。

## 例子二：A5 的完整多阶段事务任务

这是一个完整任务，不是把以下每个动作单独算成一轮：

1. 提交批次 `[商品 X 扣 5，缺失商品 Y 扣 1]`，要求整批拒绝，X 也不能改变。
2. 留出变体将缺失 ID 换成库存不足，仍须整批回滚。
3. 提交合法批次 K：X 扣 3、Z 扣 4，revision 只增加一次。
4. 将 records/cache/revision/receipts 写入 SQLite 检查点，关闭连接。
5. 启动另一个 Python 子进程，从检查点恢复，再提交相同 K；不能重复扣减或增加 revision。
6. 用 K 提交不同 changes，要求 idempotency_conflict 且无副作用。
7. 另一个租户也使用 K，必须独立执行，不能复用前一租户回执。
8. 使用新 key、过期 expected_revision 写入，要求 revision_conflict 且无副作用。
9. 检查最终库存、revision、错误码；记录两个不同 worker PID 和分阶段日志。

该任务同时依赖 A1 的租户边界、A3 的版本/缓存一致性和 A4 的读取策略。只添加回执逻辑，
无法修复前半批已写入的原子性问题；只复制 records，也无法解决跨重启重复提交。

## 单任务流程

仍使用同一套四角色循环图，但 V2 增加明确的 inspect 能力：

```text
Retriever 获取源码/文档及有效记忆
    ↓
Planner → Executor.inspect（模块依赖、SQL 租户行数、配置版本、历史请求日志）
    ↓
运维：deploy → probe 多请求复现场景
研发：patch 多文件原子补丁 → probe 多请求复现场景
    ↓
Verifier 公开回归 → 不通过则有界重规划/修复
    ↓
独立留出验收 → 通过才提交共享记忆
```

Qwen 的补丁接口为：

```json
{
  "tool": "patch",
  "diagnosis": "简短的跨模块原因",
  "patches": [
    {"path": "repository.py", "old": "精确原文", "new": "替换内容"},
    {"path": "cache.py", "old": "精确原文", "new": "替换内容"}
  ]
}
```

所有编辑先在副本中检查。任一文件不匹配、路径越界、签名或允许语法不合法，整批不提交。
通过后写入内容寻址 release 目录，并原子替换唯一权威 bundle manifest，实现候选版本整体
切换；部署也按完整 bundle 切换。每次源码变更后都必须重新 probe。

## 业务状态和关联记忆

任务序列持久保留代码、release、SQLite 基础数据、版本、公开执行日志和共享记忆。
B3 之前新增一条商品并提升 data_revision；B4 之前提升 policy_revision、改变控制面配置。
这些是声明的业务输入变化，不是注入修复或答案。

复现/验收场景使用基础数据库的隔离分叉，场景内连续操作真实更新缓存、库存和回执。
重复 probe 从相同初始场景开始，避免第一次坏更新污染第二次复现；独立评测不会改变后续
业务世界。历史契约回归使用基线配置，新配置变更由策略场景单独验证。

共享记忆有两层有效性：

- 源码快照：项目和精确 bundle 哈希一致才可复用。
- 运行经验：data_revision / policy_revision 也需一致。若运行上下文变化，保留有效代码
  快照但过滤旧 observations，明确要求重新 inspect，不把旧日志当作当前现场证据。

对应 `context_filtered_reuse` 计数。完整参考链中，B3 和 B4 各触发一次过滤。

## 独立验收

每种 incident 有两个公开场景变体。后续任务会重跑全部前序契约。留出场景使用不同租户名、
商品 ID、库存值、重复名称分布与归档分布；ID/stock 不由模型任务 ID 推导。
原子事务变体包括缺失 ID 和库存不足两种失败原因。

验收根据实际执行输出、最终记录/版本、缓存计数和错误码检查，不根据模型总结判断成功。
worker 运行异常不会被当作有效的“复现成功”。模型自由总结留在报告，记忆事实摘要仍由
通过的可执行目标生成。

子进程只接受项目白名单模块导入、固定函数接口及受限 Python 语法；有 CPU、内存和墙钟
限制。它不是任意 CodeAct 或容器安全沙箱，也没有真实 HTTP、并发压测或主机崩溃恢复保证。
事务重启检查针对子 worker，检查点由父运行时保存/恢复，不是 SIGKILL 分布式事务实验。

## 运行与展示

在独立项目根目录：

```bash
conda activate latentmas

# CLI 现在默认 project；显式指定便于复现
python -m tools.run_devops_workload --backend fixture --workload project --mode all --max-steps 20

# 使用真实模型；较长的多文件补丁给 Executor 2048 token，其他角色 768
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python -m tools.run_devops_workload \
  --backend qwen --workload project --mode protocol_vector_memory \
  --device cuda:0 --embedding-device cpu --max-steps 30

# 保留旧版单函数任务
python -m tools.run_devops_workload --workload service --mode all

python -m tools.serve_ui --backend fixture --device cuda:0 --embedding-device cpu --port 8765
# 浏览器 http://127.0.0.1:8765/devops，默认选择 V2
```

API 通过 `GET /api/devops/tasks?workload=project` 获取任务；创建任务时传 `workload: project`。
为兼容旧客户端，不带 workload 的 API 仍为 service。页面提供项目文件选择器，查看不同
模块的候选/部署代码，以及修改文件数、业务操作数、日志和上下文过滤记录。

报告位于 `results/devops/{backend}-{id}/{mode}/`。V2 快照为 `A1-candidate.json` 等文件 bundle；
实际模块在 `world/releases/{hash}/`，基础数据为 `world/catalog.sqlite`，公开日志为
`world/observations.jsonl`。旧 `.py` 快照和新 `.json` 快照均可用
`python -m tools.evaluate_devops_report ... --output ...` 独立复核。

新计数：`patch_files` 为累计实际发生内容变化的文件次数；`probe_operations` 为显式 probe
内部的业务操作数。工具调用数与业务操作数分别统计；回归/留出验收另外执行的操作不混入
该计数。源码上下文复用仍不等于物理磁盘 IO 减少，也不保证记忆模式更快。

## 当前验证范围

完整 fixture 链可以通过十轮及快照独立复核：30 次工具调用、104 个显式 probe 业务操作、
累计 11 次文件修改、9 次跨组有效复用、2 次运行上下文过滤。fixture 使用参考配方和固定
向量，只验证环境/机制，不是模型能力数据。

最新五模式 fixture 报告位于 `results/devops/fixture-67565a51/`，五种模式均通过 10/10；
共享模式的 `regraded.json` 独立复核通过 10/10。完整自动化测试 94 项通过，包含跨模块局部
修复不足、原子补丁失败回滚、运行异常不冒充复现、快照复核和 V2 HTTP/文件查看流程；
Python 编译与 JavaScript 语法检查通过。没有真实浏览器截图验证。

第一次真实 Qwen 开发试跑保存在 `results/devops/qwen-5da153a8/protocol_vector_memory/`：
B1 通过，A1 在 20 步内未通过跨模块修复，后续八轮阻塞，27 次模型调用。该次试跑发生在
后续日志增强和 Executor token 上限提高之前，不是最终配置的盲测或稳定成功率。
旧 V1 的 10/10 记录不能用作 V2 已解决的证据。
