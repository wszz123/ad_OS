# 独立项目部署

## 系统与依赖

- 通用Linux；Python 3.10。
- LangGraph 1.0.10、SQLite checkpoint 3.1.1。
- 真实模式需PyTorch、Transformers和本地模型权重。
- 无需vLLM；LangGraph InMemoryStore不提供跨进程记忆持久化。

```bash
conda create -n coop-prototype python=3.10 -y
conda activate coop-prototype
python -m pip install -r requirements.txt
```

在公司网络/本机代理环境下安装可临时指定pip代理：

```bash
python -m pip install -r requirements.txt --proxy http://127.0.0.1:1081
```

代理只用于依赖安装。真实模型通过`local_files_only=True`离线加载。不要将token、
密码或代理认证信息写入requirements或Git文件。

## 模型路径与设备

默认本机路径：

```text
/home/Users/SharedFile/Models/Qwen/Qwen3-8B
/home/Users/SharedFile/Models/Qwen/Qwen3-Embedding-4B
```

部署到其他Linux主机通过CLI参数覆盖路径。两张GPU空闲时可将生成/embedding分置于
cuda:0/cuda:1；若单卡有足够显存，也可共卡；RAM充分时可令embedding使用CPU。
Qwen3-Embedding-4B在CPU速度较慢。报告分别记录权重dtype、向量输出float32和状态传输float16。

## 环境诊断

```bash
python -m coop doctor --backend fixture
python -m coop doctor --backend qwen \
  --model-path /models/Qwen3-8B \
  --embedding-path /models/Qwen3-Embedding-4B
```

Doctor不载入权重，检查Python、声明依赖、模型配置/tokenizer/权重分片、GPU显存和1081端口。
READY表示静态检查就绪，不保证实际显存、代理协议或模型生成一定可用。真实加载由demo验证。

## 运行

### 快速mechanism fixture

```bash
python -m coop demo --backend fixture --output-root results/demo
```

它运行相同的四角色图、计算器、协议、记忆、状态/checkpoint路径，但Planner用公开意图
规则、embedding用2D主题示意向量、token计数用空格切分。它只验证接线，不是LLM结果或
性能证据。输出会标注`scripted_fixture`。

### 本地Qwen完整演示

```bash
python -m coop demo --backend qwen \
  --device cuda:1 --embedding-device cpu \
  --model-path /models/Qwen3-8B \
  --embedding-path /models/Qwen3-Embedding-4B \
  --cpu-threads 8 --timeout 1800
```

第一子进程共享加载一次Qwen生成/embedding模型，运行text/protocol/vector/memory四模式各十轮，
有界纠错最多2次，SQLite优化profile和batch语料初始化开启。结束后模型进程退出，再分别启动
状态checkpoint和纠错ledger的SIGKILL恢复worker，避免第二份8B实例争抢显存。
`--skip-recovery`只运行任务对照。每一步超时后只终止CLI自己创建的进程组并保留日志。

### 汇总既有报告

```bash
python -m coop summarize --results-dir results --output results/evidence_summary
# 也可限定输入
python -m coop summarize --report results/demo/qwen-RUN/workload.json \
  --report results/demo/qwen-RUN/state_recovery.json --output results/demo/qwen-RUN/summary
```

摘要保留失败、按报告类型标注fixture/真实模型，不合并不同task的准确率，不根据局部时间
推断通用加速比。

## 结果目录与持久化

每次演示创建唯一`results/demo/{backend}-{run_id}/`，包括：

- `demo.json`：执行命令、版本、步骤状态、日志、时长。
- `workload.json`：四模式指标及逐任务公式、计算、答案和引用。
- 每模式SQLite语料、任务记忆、状态包、checkpoint、纠错ledger。
- 每模式协议frame/token/trace及memory snapshot。
- `state_recovery.json`、`repair_recovery.json`和对应worker logs。
- `summary.json`及`summary.md`。

真实数据记忆和图状态存SQLite。fixture演示可重复但不使用模型能力。
SQLite `put_many`仅用于语料初始化；任务产生的共享记忆按任务提交。
InMemoryStore只在单进程存活；将graph checkpoint设为SQLite不等于记忆Store也持久化。

## 独立源码包

```bash
python -m coop package --include-reports --output results/coop-prototype.zip
```

ZIP自带`PACKAGE_MANIFEST.json`，含每文件大小和SHA256；排除权重、数据库、日志、缓存、
运行ZIP和编译缓存。验证解包目录：

```bash
python -m tools.verify_source_package results/coop-prototype.zip \
  --output-directory /tmp/coop-package-check
```

现有latentmas Conda环境已实测。`requirements.txt`给出固定的原型依赖；若要在完全新环境
复现CUDA，请先按PyTorch官方渠道安装对应CUDA wheel，再安装requirements。
