# 研发—运维双组连续任务：可执行首版

本文说明 V1 单函数机制练习。当前 CLI 和页面默认已升级为 V2 多文件有状态项目，见
[`CATALOG_PROJECT_V2.md`](CATALOG_PROJECT_V2.md)。运行本文任务请显式传 `--workload service`。

## 目标与证据边界

对应 `proj_requir.md` 第 8–16 条：四角色协作、协议对照、非文本状态使用、两组各五轮、
跨组共享记忆、十轮执行及独立验收。入口为 `tools/run_devops_workload.py`。

本版是**自建小型 catalog-search 服务练习**，不是 AppWorld、The Agent Company 或真实生产
数据集；它先验证新架构和研发/运维闭环，不能代表复杂真实项目的难度。它实际执行 Python
服务函数、修改候选代码、复制部署、运行公开回归和独立留出输入。服务通过隔离的 Python
子进程接收请求，没有常驻 HTTP 服务、真实流量或在线灰度。

fixture 决策器包含明确的参考修复配方、固定四维示意向量，只作接线与故障边界验证。
Qwen 后端让模型生成下一步、实际补丁和验证总结，不把参考配方、留出输入或评分结果放入
模型上下文。Qwen 接口已接入，但不能用 fixture 成功率代替真实模型运行结果。
本地 Qwen 的最终开发运行和独立复核均为 10/10；历史失败与口径见
[`DEVOPS_VALIDATION.md`](DEVOPS_VALIDATION.md)，该结果不是外部 benchmark 或通用加速证据。

## 两组连续任务及关联

两组操作同一个服务的 candidate 和 deployed 两个版本。每轮都是完整、可独立验收的请求；
运维任务的交付是版本部署、故障复现证据和先前修复检查，研发任务的交付是新修复及回归。

| 顺序 | 任务 | 独立交付与验收 |
| --- | --- | --- |
| 1 | B1 运维 | 部署初版；复现查询首尾空白导致的漏查，记录请求、实际结果与版本 |
| 2 | A1 研发 | 修复空白处理；公开和留出输入均符合要求 |
| 3 | B2 运维 | 部署 A1；确认空白修复有效，复现 offset=0 被默认值覆盖 |
| 4 | A2 研发 | 修复分页偏移；保持空白行为正确 |
| 5 | B3 运维 | 部署 A2；确认前两项修复，复现负 limit 返回非空结果 |
| 6 | A3 研发 | 将 limit 限制在 0..100；保持前两项正确 |
| 7 | B4 运维 | 部署 A3；确认前序修复，复现字符串 false 被当作 true |
| 8 | A4 研发 | 修复 active 字符串解释；保持前序行为正确 |
| 9 | B5 运维 | 部署 A4；确认前序修复，复现 Unicode 大小写匹配缺失 |
| 10 | A5 研发 | 修复 STRASSE/ Straße 匹配；全部公开及留出检查通过 |

组内链：`A1 → A2 → A3 → A4 → A5`、`B1 → B2 → B3 → B4 → B5`。
跨组链：`B1 → A1 → B2 → A2 → … → B5 → A5`。

业务状态依赖和共享记忆分别处理：即使关闭记忆，候选代码、部署版本、前轮合法产物仍然
保留；打开记忆只是额外提供此前验证的源码快照、诊断观察和经验摘要。失败会阻塞依赖任务，
不会注入标准补丁以凑满成功十轮。A5 完成代表最终候选已验收，不代表它已经上线。

## Agent 与 LangGraph

三种模型角色共享一份 Qwen 权重、分别调用；Retriever 是确定性检索角色。

- Planner：读任务、证据和历史经验，选择 retrieve / execute / verify。
- Retriever：检索项目指南和历史记忆，校验来源及源码版本；将接受的证据交给其他角色。
- Executor：根据 Planner 选择生成 deploy / patch / probe 调用。
- Verifier：读公开检查结果和执行证据，给出结论与摘要。

```text
START → initialize → retrieve → plan
                                ├─ retrieve ────────────┐
                                ├─ execute → tools ─────┤
                                └─ verify               │
                                    ├─ failed → plan ←──┘
                                    └─ verified → finish → END
```

`plan` 对路由结果做确定性检查；工具按 task.group 限制：ops 不得 patch，dev 不得 deploy。
当前能力约束随已完成动作变化：部署成功后不重复部署，每次修改后必须先 probe 当前版本，
当前复现/修复目标满足后进入验证。模型返回 `route=probe/patch/deploy` 时显式映射到 execute，
原始返回保留在 trace 中；不替模型生成补丁或更改验收目标。
旧任务观察不能代替当前任务 probe；未取得当前证据时不能跳到 verify。无效规划会收到错误
观察并在剩余预算内重规划。公开验证失败时重新开放修复分支，避免只通过单个示例就结束。
probe 运行时错误也带当前版本、目标和请求回传；研发可据此重新修复，不会被困在重复 probe。
公开/留出验收把运行异常记为失败，而不是让验收器抛错丢失整轮结果。
最多 `--max-steps` 步（默认 12，按检索/规划/执行/验证节点计数）；耗尽后明确失败。
Verifier 的模型判断必须与公开回归共同通过，不能用“模型认为完成”跳过实际检查。
公开检查分别记录 `behavior_correct`（服务行为是否正确）和 `passed`（是否满足当前任务）。
运维当前目标为复现故障，故该项可以 behavior_correct=false、passed=true；研发修复要求两者
均为 true。这样不会把成功复现误标成修复验收失败。
公开验证还覆盖已修复功能的默认分页、limit 上界和布尔 false；留出验证使用不同记录和输入。

任务调度器在图外逐轮运行。每轮重建通信历史；持久记忆和业务环境跨轮保留。当前 CLI 要求
新的空输出目录，不实现进程崩溃后续跑；图构造接口可接 LangGraph checkpointer，但本版没有
承诺跨进程的模型预算、状态包和工具副作用原子恢复。

## 非文本状态的使用

规划端准备检索 query，生成向量并用 `StateExchange` 编码成 float16 二进制包，控制消息
只传 state ID。Retriever 解码后直接用于项目指南及历史记忆的向量检索，不重新计算 query
embedding。无记忆向量模式仍用该向量查项目指南。结果目录保存 `.bin` 和 dtype/shape/
checksum 等元数据 JSON。

fixture 固定向量仅检查编码与检索路径；真实语义检索必须使用 Qwen embedding 后端。
此版状态交换是同进程状态仓库，不是 IPC/共享内存性能实验。

## 共享记忆与验收隔离

记忆包含赛题规定的 ID、来源 Agent、时间、主题、摘要，并附带：

- run ID、项目 ID、来源任务及研发/运维组；
- 当前源码 SHA256、源码快照；
- 执行观察、验证标记和经验摘要。

共享记忆的事实摘要由程序依据已通过的任务目标生成，标明运维“复现但未修复”或研发
“候选修复但未部署”。模型自由总结留在原始报告/trace 中供审计，不自动升级为已验证事实，
避免“成功复现”被错误总结成“已经修复”后跨组传播。

检索必须匹配当前 run、项目和精确源码版本，且源码快照自身哈希一致。过期或损坏记忆拒用；
当前版本下优先最近记忆。部署诊断不能代替对当前部署的重新 probe。

图结束先产出记忆候选；调度器运行独立留出评测后才提交记忆。留出评测不反馈给 Agent，
不驱动重试；失败即阻塞后续依赖任务。当前 ops 的独立验收检查真实部署及复现轨迹，dev
额外检查更换记录与参数后的结果。仍需更多负例才能作为充分的功能正确性保证。

当前直接可观测复用是：避免再次将源码加载为 Agent 上下文，并向 Planner 提供已有观察和
经验。它不是免除源码校验：SHA256 校验仍读取源文件，因此 `source_context_reads` 和
`avoided_source_reads` **不是物理磁盘读次数/字节数**。fixture 不会证明语义推理或诊断加速。

## 五模式对照

| mode | 通信 | 向量 | 记忆 |
| --- | --- | --- | --- |
| text | 现有长度标记文本协议，携带真实当前任务历史 | 无 | 无 |
| protocol | 结构化消息 | 无 | 无 |
| protocol_vector | 结构化消息 | 有 | 无 |
| protocol_vector_group_memory | 结构化消息 | 有 | 仅来源同组 |
| protocol_vector_memory | 结构化消息 | 有 | 双组共享 |

各模式从同一源码和部署快照开始，工作区、记忆和报告隔离。最后两种模式专门测组间共享的
作用。文本基线不是自由自然语言解析的 MAS；不能将结果外推成对所有文本多 Agent 的优势。
传输采用同一输入/输出字段，没有给文本模式额外生成填充内容。

记录：消息数、UTF-8 字节和字符、状态次数和 payload、任务用时、独立评测用时、工具调用、
模型调用及 prompt/output token、记忆候选/接受/拒绝/跨组使用、验收后有效复用。
接受查询率为接受次数/记忆查询次数，不是检索候选的 precision。计时不含模型加载、图编译、
初始化项目指南索引和结果文件归档；报告不得把这些成本当作零。

fixture 模型调用和模型 token 为 0；其通信字符/字节可复核，但不当作真实 Qwen 推理证据。
共享记忆带来的额外查找与摘要传输也计入成本，可能更慢、更长，不预设必然加速。

## 运行

在独立项目根目录运行：

```bash
conda activate latentmas

# 五模式，每种两组各五轮；每次生成唯一实验目录
python -m tools.run_devops_workload --backend fixture --workload service --mode all

# 只验证双组共享闭环
python -m tools.run_devops_workload --workload service --mode protocol_vector_memory

# 真实本地模型；加载时间/显存与任务成功率需实际验证
python -m tools.run_devops_workload --backend qwen --workload service --mode protocol_vector_memory \
  --device cuda:0 --embedding-device cpu --max-steps 30

python -m unittest tests.test_devops_workload -v
```

目录 `results/devops/{backend}-{id}/{mode}/` 包含 report.json、逐任务通信 trace、二进制状态
及元数据、候选与部署源码快照，以及 memory.sqlite。report 中有任务分组、状态、检查结果、
复用 ID、拒用 ID、逐轮指标和汇总。

可以不重跑模型，单独复核已保存的快照（替换实际实验路径）：

```bash
python -m tools.evaluate_devops_report results/devops/实际实验ID/protocol_vector_memory/report.json \
  --output results/devops/实际实验ID/protocol_vector_memory/regraded.json
```

复核先检查快照与原记录的版本哈希，使用当前验收器重执行；输出另存，原始结果不覆盖。
如果前置任务在新检查中失败，下游结果不再被计作有效成功。该复核不产生新的模型调用，
不能替代使用最终版本代码重新运行实验。`python -m coop summarize --report <report.json>`
也支持汇总研发—运维报告，并按 fixture/真实模型标记证据。

工具执行采用限制语法、受限 builtins、`python -I` 子进程和三秒超时。它只允许小型搜索函数
补丁，不支持任意 Python CodeAct，也不替代容器级文件系统/资源隔离。

## 浏览器运行与检查

```bash
python -m tools.serve_ui --backend fixture --device cuda:0 --embedding-device cpu --port 8765
```

打开 `http://127.0.0.1:8765/devops`，也可从原有首页进入“研发—运维闭环”。选择五种模式之一、
步数预算和后端，运行十轮。Qwen 会使用页面指定的生成/向量设备，加载耗时另外等待。
页面展示双组任务状态、跨组有效复用、传输字节、模型调用；点击已结束任务查看工具观察、
独立验收、记忆来源/拒用 ID、候选/部署代码和协议 trace。fixture 范围在页面明确标注。

API：

- `GET /api/devops/tasks`：两组任务及依赖。
- `POST /api/devops/runs`：传 `mode`、`max_steps`，202 返回 job_id，使用当前引擎。
- `GET /api/devops/runs/{job_id}`：队列/运行/结束状态和逐轮报告。
- `GET /api/devops/runs/{job_id}/tasks/{A1..A5|B1..B5}`：已执行任务的 trace 和代码。

每次运行独立工作区和记忆；任务在后台线程执行，模型访问与原有成本任务、引擎切换串行。
页面通过轮询显示真实逐轮状态，重载页面可重新连接当前进程中的 job。索引在服务器进程内，
重启后旧 job API 不恢复，磁盘报告仍保留在 `results/devops-web/{job_id}/`。

## 下一阶段扩展

将 `DevOpsWorld` 的工具/验收接口替换成固定版本真实软件仓库和容器部署环境；保留本版为
循环图回归测试。新增日志、配置、依赖、跨文件修复和新版本失效任务，并增加与故障相关性
无关的负例。真实项目的预置故障及任务组合应标注为自建扩展，原始代码/Issue 来源另记。
