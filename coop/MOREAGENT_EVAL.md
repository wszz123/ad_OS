# MoreAgent 医学 50 题：原始 HF 方法对照

## 输入与隔离

用户提供 `moreagent_50_questions.json`，schema 为
`moreagent.dataset-presentation-summary/1`。原文件用途是展示汇总，包含题干、选项、答案、
解析、候选关联与主题提示；不能把整条 JSON 当作模型输入。

`tools/moreagent_data.py` 显式转换：

- 模型只接收英文 question 和 A–D options，保留原顺序。
- question_id 只用于记录；传给原方法对象的 gold、solution 均为空。
- evaluation_only、中文摘要、key cues、关联题号和解析不进入模型或初始记忆。
- 标准答案独立用于评测；label 与 answer_text 一致性、序号、重复 ID、四选项均先校验。
- 记录原文件 SHA256、转换后的公开输入 SHA256，不修改用户原文件。

公开输入 digest 定义为按题序序列化、去末尾换行的 canonical JSONL；汇总器同时检查实际
JSONL 文件（含末尾换行）、独立 labels 文件和原始数据提取结果完全一致，再重算评分。

数据分布：贫血 17、肾脏 17、甲状腺 16；30 道原 MedQA 题、20 道项目自编衍生题。
前 30 题用于过项目开发，自编题未经独立医学审题，所以这不是官方、未见 MedQA 测试。
候选关联仅是设计注释，不是实际记忆召回证据，也不是同一患者的连续病程。

## 方法与配置

运行脚本 `tools/run_moreagent_eval.py` 是**父研究仓库的评测桥接入口**，复用：

- `methods/baseline.py`：单 Agent。
- `methods/text_mas.py`：Planner → Critic → Refiner → Judger，文本上下文。
- `methods/latent_mas.py`：前三角色 latent rollout，Judger 接收 KV 后输出最终答案。

这不是把检索 embedding 改名为 latent，也不是 V2 研发—运维任务的 hybrid backend。
模型执行需要父目录 LatentMAS 研究代码；独立源码包只提供该桥接脚本和数据校验能力，
不会自动带上整个父研究仓库。没有修改父目录研究源码。

首轮对照：本地 Qwen3-8B，HF、batch=1、greedy、关闭 Qwen thinking、sequential、
latent_steps=4、完整任务内 KV，不跨题复用 KV，不启用 RAG 或长期记忆。
每次文本生成的上限相同，但 TextMAS 有四次文本生成，LatentMAS 有三次 latent rollout
加一次最终文本，不能当作相同 FLOPs 或相同总体计算预算。

为先得到可解析的快速试测，`--response-style concise` 给三种方法的所有角色统一附加：
可见回答最多 60 词；需要给最终答案的角色将 boxed 选项放在开头，随后简短说明；规划/
评审角色仍不提供最终答案。这改变了原提示词，因此结果必须标为短答设置。
`--response-style original` 保留原提示词，不混合汇总两个配置。

原实现关闭 realign 时仍计算稠密解后替换为单位矩阵，容易因无效临时分配 OOM。桥接层
采用等价的单位矩阵和分块 FP32 embedding 行范数均值，跳过被丢弃的 solve；配置为关闭
realign，归一化准备时间单独记录。微型张量测试检查归一化一致性；没有绕过已启用的
真实 alignment。运行报告保留此设置，不声称初始化与原实现逐指令一致。

## 运行

从 `competition-prototype` 目录执行：

```bash
conda activate latentmas

# 仅校验/转换模式，不加载模型
python -m tools.run_moreagent_eval --validate-only

# 首题路径检查
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python -m tools.run_moreagent_eval \
  --max-samples 1 --response-style concise --max-new-tokens 256 --latent-steps 4 --device cuda:1

# 五十题，三个方法固定顺序逐题配对
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python -m tools.run_moreagent_eval \
  --max-samples 50 --response-style concise --max-new-tokens 256 --latent-steps 4 --device cuda:1

# 对保存结果重新评分并生成表格，替换实际实验目录
python -m tools.summarize_moreagent_eval results/moreagent/实际实验ID/report.json
```

入口可用 `--methods baseline text_mas latent_mas` 选择方法，用 `--model-path` 指定本地模型。
每次生成独立 `results/moreagent/moreagent-<id>/`，逐题原子更新 report.json。

## 输出与评分

- `questions.public.jsonl`：白名单转换后的题干/选项。
- `labels.evaluation-only.json`：评测答案及分组，绝不作为 Agent 输入。
- `report.json`：配置、数据指纹、逐题预测、耗时、token、forward 与 KV 指标。
- `01-baseline-trace.json` 等：各角色实际 prompt、输出、token ID 和 latent 协议统计。
- `verified_summary.json` / `summary.md`：独立读取源数据与原始预测重新评分后的表格。

取最后一个合法 boxed A–D；兼容 boxed text/roman 格式和单字母直接答案。不从推理段落中
任意寻找 A/B/C/D。未解析答案按错误计，运行错误保留在分母。不能因为解析失败就删除样本。
按原题/自编题、三个主题分别统计，并报告两种方法同题都对、仅一方对、都错的配对数。

token 为实际生成序列长度（含 EOS）；中间文本 token 是解码输出重新分词，单独标明。
forward hook 计数真实 HF 模块调用；latent recurrent steps 与可见生成 token 分开统计。
推理耗时同步 GPU 后记录，模型加载、latent 初始化、结果审计/写文件分别记录。
KV tensor bytes 为交接时的逻辑张量规模，同进程引用不等于这些字节真实上网传输。

这是一轮固定顺序开发试测，GPU 可能与其他任务共享；不能根据一次结果推广普遍加速、
显著性或临床能力，也不能据此声称跨题记忆有效。本轮没有加载关联提示、预置知识答案或
根据 gold 选择/筛掉模型输出。

## 2026-10-21 首轮结果

完整运行：`results/moreagent/moreagent-7986f34d/`。50 题 × 3 方法全部完成；原始数据指纹、
白名单公开输入、独立 labels 文件与原始回答重新评分均通过核查。短答配置为上述 concise、
256 token、4 latent steps；Qwen3-8B、cuda:1、greedy。没有 RAG 或跨题记忆。

| 方法 | 全部 50 题 | 原 MedQA 30 题 | 自编 20 题 | 推理总秒 | 实际生成 token |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 42/50，84% | 22/30，73.3% | 20/20，100% | 59.99 | 2,863 |
| TextMAS | 45/50，90% | 25/30，83.3% | 20/20，100% | 362.00 | 17,278 |
| LatentMAS | 43/50，86% | 23/30，76.7% | 20/20，100% | 99.88 | 3,685 |

三种方法无效答案数和运行错误数均为 0。TextMAS 与 LatentMAS 都答对 41 题，仅 TextMAS
答对 4 题、仅 LatentMAS 答对 2 题、都错 3 题。LatentMAS 对比单体多答对 1 题，耗时更多；
对比 TextMAS 少答对 2 题，推理总时长约减少 72%，可见生成 token 约减少 79%。
这是本配置的一次观察，不是同等 FLOPs 下普遍优越的结论。

自编 20 题三种方法全对，当前不能有效区分方法；不能只看合并后的高准确率。文件的关联
设计也尚未转化为实测记忆收益，因为本轮禁用了跨题记忆。

LatentMAS 的 150 次 rollout 共执行 600 个 latent recurrent steps；逻辑 KV 交接张量累计
17,861,345,280 字节，控制帧 93,492 字节。KV 在同进程中引用交接，前者不是实测网络流量。
详表、配对题号和逐题完整输出见本次 `summary.md`、`verified_summary.json`、`report.json`
和各题 trace。启动/审计另计；整个驱动进程约 540 秒。

另有原提示词单题试跑 `moreagent-40f03eab`：512 token 下 baseline 未在预算内输出可解析
答案，TextMAS/LatentMAS 给出正确选项。它只用于发现输出预算问题，不与上述 50 题结果混算。
