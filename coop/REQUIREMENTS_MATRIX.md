# 对照 proj_requir.md 的验收状态

原始要求以仓库根目录 `proj_requir.md` 为准。这里区分代码覆盖和实验边界。

| 原文要求 | 当前可运行实现/证据 | 状态与边界 |
| --- | --- | --- |
| 至少 3 Agent、3 类角色、多步骤复杂任务（第 8 行） | LatentMAS Planner/Critic/Refiner/Judger，加Retriever/ToolExecutor图节点 | 两组真实Qwen检索+工具计算任务；过滤来源、bounded calculator、citation均记录 |
| 动作、参数、结果、能力描述及发现（第 9 行） | `protocol.py`、能力注册/发现与版本校验测试 | 已实现本地适配器，尚无网络握手 |
| 纯文本/结构化同任务对照（第 10 行） | `text_transport.py`、四模式 CLI、保存真实消息帧 | 标签文本编解码基线，不是 LLM 自然语言解析基线 |
| 非文本状态的生成、传递、接收、使用（第 11 行） | 原生四角色LangGraph；safetensors/SQLite；AF_UNIX socket KV传输 | Qwen真实socket wire 160,307,790 bytes，张量逐位一致；远程网络仍未测 |
| 统一记忆及 ID/Agent/时间/主题/摘要（第 12 行） | `memory.py` schema、来源快照和任务memory；LangGraph state/checkpoint保存refs与metrics | SQLite跨进程持久；LangGraph InMemoryStore仅进程内，缓存只复用事实不复用旧答案 |
| 关键词/标签/语义检索及不同 Agent 复用（第 13 行） | 三种检索 API，SQLite/LangGraph Store 适配及同任务后端对照 | 两后端结果一致；LangGraph InMemoryStore 不持久，其他服务/持久 provider 尚未验证 |
| 至少两组关联连续任务（第 14 行） | 简单egg/bus十题；复杂server-cost/power-cost十题两轮反序对照 | 最终两组复杂任务四模式均10/10；早期预算/提示配置负结果另存 |
| 消息、文本开销、状态规模、耗时、复用率及性能比较（第 15 行） | `run_cost_workload.py`、audit JSON、逐任务时间/调用/引用 | 已区分通信/模型 token 与状态元数据；如实保留负结果 |
| 五个系统模块、稳定至少 10 轮（第 16 行） | 五层 Adapter，正常四角色十轮全通过；状态与 Planner 预算 SIGKILL 恢复 | 节点内预算跨进程不重置、接受/耗尽零调用重放已验证；并发租约和其他节点 side effect 仍待验证 |
| 完整源码、设计/部署/实验文档、演示视频（第 17 行） | 统一CLI、源码ZIP/清单、部署文档、实验工件、录制脚本 | 快速/真实演示已通过；正式提交审核与视频仍待完成 |
| CodeAct（第 18 行，加分） | 当前为 AST 算术工具，拒绝任意代码执行 | 未实现 CodeAct，不将计算器描述为代码沙箱 |

## 对评分项的当前证据

- **通信效率 25 分**：compact控制帧降低约5.2% tokenizer tokens/19.2% bytes；但AF_UNIX每十题传约1.594GB KV，不能声称总通信低于TextMAS。外围成本图45.42%是另一工作流结果。
- **状态传递 20 分**：原HF循环与原生LangGraph均通过Protocol执行KV handoff；KV/checkpoint新进程恢复已验证，Planner full快照24,041,728bytes，三份图快照累计160,303,960bytes。网络wire待完成，embedding仅辅助对照。
- **记忆复用 20 分**：两组复杂任务每轮short-circuit 8/10题，wrapper calls 40→8，保存32次；仅限两个严格任务合同。
- **系统完整性 20 分**：四角色主线已接Retriever/ToolExecutor、IPC、统一SQLite资源文件和verified short-circuit；跨连接事务仍分开提交，并发lease和远程worker未实现。
- **初始化可靠性/开销**：SQLite 原子批量事务与 savepoint 已验证，137条语料提交137→1，未降低同步级别，工作流结果一致。
- **实验验证 15 分**：两轮反序真实Qwen复杂任务四模式均10/10；分别计数Transformers forward、KV bytes、AF_UNIX wire和模型调用。任务是固定synthetic contracts，尚非泛化证明。
