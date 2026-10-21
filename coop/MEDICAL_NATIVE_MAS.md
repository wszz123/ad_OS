# 子仓库原生医学 MAS 适配

## 这次适配的是什么

入口为 `tools/run_medical_workload.py`，图与记忆实现分别位于 `coop/medical_graph.py`、
`coop/medical_memory.py`。它直接使用子仓库的 `ProtocolAdapter`、`TextAdapter`、
`StateExchange`、`MemoryAdapter` 和 `QwenCausalAdapter`；没有导入父目录的
`methods/` 或 `models.py`，不是在网页里模拟 LatentMAS。

任务适配为医学选择题，单题 Agent 可见字段只有 `question_id/question/options` 白名单中
实际下发给角色的是 question 和 A-D options；答案、中文摘要、提示词线索、解析、候选关联
题号均不进入模型、向量、语料、跨题记忆或首发请求。独立评分标签保存在单独文件，由评测端
在 Agent 完成后比较最终选项。

## 子仓库图结构

```text
START → Planner → Retriever → Executor → Verifier → END
           │           │          │            │
           │           │          │            └─ 校验选项 schema、来源 ID、条件复核
           │           │          └─ 根据病例与检索材料形成候选答案
           │           └─ 五条通用检查清单 + 可选的历史病例模型笔记
           └─ 主题、搜索 query、最多四项检查和原题逐字摘录
```

这是每题的四个角色调用，不是父仓库四个生成式 Planner/Critic/Refiner/Judger：

- Planner、Executor、Verifier 调用模型。
- Retriever 根据标签/语义检索通用工作流清单与在线记忆。
- 评分器不在图内，也不被模型调用。
- schema 解析失败最多有界重试；最后失败记为 error，不能猜答案。

generic corpus 只有“比较鉴别诊断”“重新核对年龄与病史”等五条**非医学知识**检查清单，
不含诊断结论、标准答案或教材证据。它不能替代真正的 RAG 临床证据库。

## Memory 的适配边界

医学版本不把“Verifier 接受答案”或最终选项当作真理，也没有 gold-based 准入。每题结束后，
Verifier 可以写一个短记忆笔记，字段为 `claim/applies_when/exceptions`，状态固定为
`unverified_model_note`。写入前必须提供至少一段从**该题题干逐字抽取**的原文，并记录
question_id、sequence、题干哈希、模型标识、实验 namespace 和分组主题。

读取时检查 schema 版本、实验/模型/来源题指纹、题目确实是前序题、来源摘录和摘要哈希；
错误、自引、未来题或损坏记录会拒用。被 Verifier 声明使用的每条 note 都必须提供
`condition_checks`，说明本题哪些条件已重新核对。

这些检查只能证明来源可追溯和模型完成了格式化的条件复核，**不能证明病例摘录足以支持医学
结论，也不能证明笔记正确**。所以不能将其命名为 verified medical knowledge，也不能把
“declared reuse”称为有效医学证据复用。标准答案和正确性绝不影响 memory admission。

每种 memory 模式都从独立空库开始，按输入顺序在线写入；无记忆模式仍执行全部角色，但不
检索/写长期 memory。topic group 由 Planner 根据当前公开题干自行分类，不使用评测侧 topic
字段；错分风险由报告保留。

## 五模式

| 模式 | 角色通信 | 非文本 query state | 在线 memory |
| --- | --- | --- | --- |
| `text` | 真实当前任务内请求/响应文本历史 | 无 | 无 |
| `protocol` | 结构化 JSON envelope | 无 | 无 |
| `protocol_vector` | 结构化协议 | float16 Qwen 查询向量用于 guide 检索 | 无 |
| `protocol_vector_group_memory` | 结构化协议 | float16 查询向量用于 guide/病例记忆检索 | topic 内 |
| `protocol_vector_memory` | 结构化协议 | 同上 | 跨 anemia/renal/thyroid 共享 |

查询向量由 Planner 的 topic/search query 编码，再由 Retriever 解码直接检索；它不是 hidden
state/KV 接续。Text 与 Protocol 向 agent handler 传递相同语义字段；TextAdapter 会附加真实
历史，因此 UTF-8 bytes 一般更高，这是设计中的通信消融，不是自然语言 MAS 的全面比较。

## 运行

数据文件：`moreagent_50_questions.json`。

```bash
conda activate latentmas

# 校验题目结构与来源分布，不加载模型
python -m tools.run_medical_workload --validate-only

# Fixture：检查协议/隔离/记忆流水线；固定回答 A，非推理性能证据
python -m tools.run_medical_workload --backend fixture --modes all --max-samples 50

# 本地 Qwen 50题 × 五模式；模型生成与 embedding 可放不同 GPU
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python -m tools.run_medical_workload \
  --backend qwen --modes all --max-samples 50 \
  --device cuda:1 --embedding-device cuda:0 --top-k 2 --schema-attempts 2

# 隔离一个模式检查路径
python -m tools.run_medical_workload --backend qwen \
  --modes protocol_vector_memory --max-samples 10 \
  --device cuda:1 --embedding-device cuda:0

# 对原始回答、答案和逐题顺序做独立复核及配对摘要
python -m tools.verify_medical_report results/medical-native/medical-qwen-ID/report.json
```

真实报告以 `medical-qwen-*` 保存。包含 whitelist public JSONL、独立 labels 文件、五种模式
逐题结果、角色 prompt/response、协议帧、状态包 metadata/payload、Memory 来源、重试、调用、
token 和耗时。Verifier 工具重新从源文件加载 gold，检查 public 输入/labels 指纹、选项范围、
raw JSON 和选项、记忆必须来自已完成前序任务，最后重新统计分来源和分主题结果。可把多个
完整运行结果路径一起交给 verifier 配对；运行条件需由研究者核对一致。

### 指标口径

- correctness 是选项与文件中 evaluator-only label 的逐题匹配；不是医学正确性审查。
- 无效格式计入错误；发生工具/图异常则保留 error row，仍留在 accuracy 分母。
- memory writes 是未验证模型笔记写入数；retrieval hit 是候选数/查询次数；declared reuse 是模型自报引用，不等于证据正确。
- `correct_with_declared_reuse` 是答案正确且模型声明使用记忆的描述性交集；不建立记忆的因果收益。
- KV bytes 仅适用于 Latent KV；本图为 embedding 状态。状态 payload、metadata、控制帧字节分开记录。
- forward hook 只计 Causal LM forward；Embedding 模型调用、模型加载、数据库初始化单列/排除，不能称为完整 FLOPs。
- 固定顺序单轮、同机共享 GPU，无置信区间；文本模式多生成次数与 latent/memory 模式计算不等价。

## 真实 Qwen 首轮结果（2026-10-21）

完整对照摘要：`results/medical-native/medical-qwen-combined/verified_summary.json`。
为避免同时加载 Qwen3-8B 与 4B embedding 模型挤占同一卡，实验分两次进程完成：无记忆
Text/Protocol/Vector 来自 `medical-qwen-4fe891dc/report.json`；同主题/跨主题记忆来自
`medical-qwen-05e6dfea/report.json`。两次均为 Qwen3-8B + Qwen3-Embedding-4B，同一数据
指纹、题序、greedy、role token budget、top-k=2、schema_attempts=2；生成/embedding 设备
配置不同，时延不能直接横向配对。组合表对五个已完整执行的 50 题模式按同题 ID 和标签配对。

| 子仓库原生模式 | 答对/50 | 准确率 | role/schema errors | memory 候选命中/50 | 模型声明使用 | 正确且声明使用 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Text | 42 | 84% | 1 | 0 | 0 | 0 |
| Protocol | 42 | 84% | 1 | 0 | 0 | 0 |
| Protocol + vector | 43 | 86% | 0 | 0 | 0 | 0 |
| Protocol + vector + group memory | 37 | 74% | 7 | 39 | 40 | 23 |
| Protocol + vector + shared memory | 40 | 80% | 2 | 47 | 29 | 18 |

group/shared memory 的 20 道自编题准确率分别 85%/90%；原 30 道开发用过的 MedQA 题为
66.7%/73.3%。Vector 无记忆模式为 43/50 (86%)，高于两种记忆模式。该单次运行没有显示
记忆提升总体准确率。Group 与 shared 的逐题配对：35 题都对、group-only 2、shared-only 5、
都错 8。答案正确且模型声明复用的题目分别为 23/18；这只是描述性交集，不能证明 declared
reuse 的临床有效性或因果收益。

schema/执行错误也计入 50 题分母：group memory 7 次，shared memory 2 次；无记忆 text/protocol
各 1 次，vector 为 0。准确率不能只看成功解析的完成子集。

无记忆 vector 组 150 role calls、86,460 prompt / 18,980 generated tokens、约 413.9 s；
group memory 163 calls、178,787 / 25,832 tokens、约 557.0 s；shared memory 154 calls、
171,472 / 21,145 tokens、约 459.2 s。两个记忆模式均额外检索和写入；不同 GPU 分配及
运行期间设备状态不同，以上耗时为各自任务执行时间之和，不证明记忆模式更快。Embedding
初始化/调用另计，不能只比较生成 token。

本轮为开发试测，不是盲测或独立医学评审；20 道自编题未经专家审题，不能推断临床可靠性。
记忆的 case quote 证明可追溯到题干，不证明 note 医学正确；当前没有外部权威医学知识库。

## 与父仓库 LatentMAS 的关系

该适配使用子仓库自身 Qwen adapter 和 LangGraph，不使用父仓库 `LatentMASMethod` 或 KV latent
rollout。父仓库原方法已经在同一数据上单独测试过，但不能作为本图的效果。当前实现是
Planner/Executor/Verifier 三次 JSON 生成 + 确定性 Retriever，不包含 Critic/Refiner，也没有
跨 Agent 的 transformer hidden-state/KV 接续；`protocol_vector` 仅传递 embedding query。
这使“子仓库方法”可以和父仓库原方法清楚区分。

## 结果解释限制

50题由30道开发阶段用过的 MedQA 题和20道自编、未经独立医学审题的题组成。不能称其为
独立未见测试集；新题100%可能反映合成分布容易或风格一致，必须分来源报告。原文件明确说
这是展示汇总而非运行输入，因此只有白名单转换后的 public JSONL 可送入模型。

候选关联 metadata 不能证明连续患者病程、已存储知识、被正确检索或有效复用。要展示记忆
价值，需检查具体 prior memory ID、题干原文、条件复核、模型实际使用声明和下游得分；还应
抽查模型 note，医学正确性需独立审阅。没有这些证据时只报告候选/声明命中，不报告有效医学
知识复用。
