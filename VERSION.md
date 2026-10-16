# 当前版本边界

本目录是从上层 LatentMAS 仓库隔离出的**社区赛题协作原型**版本。
内容仅覆盖 `proj_requir.md` 的系统层协作机制，不包含原论文复现实验入口。

## 纳入本版

- 五层边界：LangGraph、Protocol Adapter、Memory Adapter、State Exchange、Evaluator。
- 文本/结构化通信、embedding二进制交换、SQLite/LangGraph记忆适配。
- 四角色真实Qwen工作流、有界公式纠错、数值/单位/证据质量检查。
- 未见参数和语料扫描实验、存储/批量写入消融、SIGKILL状态与预算恢复。
- 对应测试、部署设计、实验说明和原始赛题文档。

## 排除在本版之外

- 根目录原始 `run.py`、`models.py`、`methods/`、`prompts.py` 等论文/基线复现程序。
- `coop/latent_*` 与 `verified_memory.py` 分支、LatentMAS KV cache/IPC对照工具和对应测试。
- 论文图片、MedQA/GSM8K数据集、旧Qwen示例日志、历史开发备份和所有运行结果。

这些文件仍保留在上层历史/研究仓库，不在本独立版本内；本目录代码和导入可独立运行。

## 来源

- GitHub 项目：`https://github.com/wszz123/ad_OS`
- 隔离路径：`competition-prototype/`
- 根目录 `proj_requir.md` 原样作为本版本任务要求。
