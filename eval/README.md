# DeskPilotBench 评测脚本

真实 API 模式会先执行 LLM 与 Embedding 健康预检。预检失败时，不会把连接、认证、限流或模型配置问题计为 Agent 能力失败；运行途中连续出现 3 个连接或限流错误时会触发熔断，剩余用例记录为 `skipped/api_circuit_open`。仅在调试预检本身时可使用 `--skip-api-preflight`，正式基线不应跳过。

## 版本基线

当前代码版本为 `0.9.0`，对比基线为 `0.8.0`。完整基线一次执行全部 `tests/test_*.py` 模块、163 条 DeskPilotBench 离线评测、22 条多模态离线评测，以及按简单/中等/复杂分层的 36 条文本 API 和 3 条 VLM API 用例：

```powershell
D:\broagent\.conda\deskpilot-py311\python.exe -m eval.run_baseline `
  --project-version 0.9.0 `
  --api-suite baseline_api_v0.9.0 `
  --multimodal-api-suite baseline_multimodal_api_v0.9.0 `
  --compare-baseline D:\broagent\eval\baselines\deskpilot_baseline_v0.8.0_20261006T191333+0800 `
  --change-summary "完成P0执行安全与敏感图片上云审批" `
  --change-summary "完成P1 Router单调用、多模态批处理、语义OCR向量、WAL与VLM流式"
```

该命令会真实消耗 LLM、Embedding 和 VLM API 配额，但网页资料和邮件副作用使用 fixture/mock，不连接私人邮箱，也不会真的发送邮件。执行前应配置三类模型 Key，并设置 `ALLOW_CLOUD_IMAGE_UPLOAD=true`；VLM 只接收脚本生成的合成测试图片。

首次运行前确认测试依赖已安装：

```powershell
D:\broagent\.conda\deskpilot-py311\python.exe -m pip install -r requirements.txt
```

产物目录遵循以下格式：

```text
eval/baselines/deskpilot_baseline_v0.8.0_YYYYMMDDThhmmss+0800/
  deskpilot_baseline_v0.8.0_YYYYMMDDThhmmss+0800.md  # 统一版本基线报告
  manifest.json                                      # 机器可读运行元数据
  test_modules.json                                  # 测试模块结果
  test_logs/                                         # 每个测试模块的原始日志
  offline_results.jsonl / offline_report.md          # 163 条离线评测
  api_results.jsonl / api_report.md                  # 36 条分级真实文本 API 评测
  multimodal_offline_results.jsonl / multimodal_offline_report.md
  multimodal_api_results.jsonl / multimodal_api_report.md  # 3 条分级真实 VLM 评测
  *_metadata.json                                    # 模型、数据集哈希、Git 状态等
  *_console.log                                      # 原始控制台输出
```

统一报告记录版本、开始/结束时间、Git 提交、工作区状态、Python/模型配置、数据集 SHA-256 和本版本变更。效率部分先按功能模块统计平均/P50/P95/累计响应时间、API 调用数和输入/输出/总 Token，再按 Router、Planner、Answer、Citation Repair、Reflection、Memory Extraction、Embedding 等调用阶段归因，最后给出整个评测工作负载总计。Token 只使用服务端返回的 `usage` 字段；服务端不报告时显示“不适用”。

后续版本必须更新版本号，并至少提供一条修改说明，例如：

```powershell
D:\broagent\.conda\deskpilot-py311\python.exe -m eval.run_baseline `
  --project-version 0.9.0 `
  --api-suite baseline_api_v0.9.0 `
  --multimodal-api-suite baseline_multimodal_api_v0.9.0 `
  --change-summary "将网页写文件工作流迁移到 PlanExecutor" `
  --change-summary "新增持久化 Task Ledger"
```

在新套件尚未建立时，可以继续使用旧固定套件作为跨版本可比样本；如果调整 API 用例集合，必须新增 suite 文件而不是覆盖旧文件，并在变更说明中注明。

只运行某一批评测时，`eval.run_eval` 也会自动使用规范文件名：

```text
deskpilotbench_v<版本>_<模式>_<带时区时间戳>.md
deskpilotbench_v<版本>_<模式>_<带时区时间戳>.jsonl
deskpilotbench_v<版本>_<模式>_<带时区时间戳>.metadata.json
```

## 快速运行

在项目根目录执行：

    .\.conda\deskpilot-py311\python.exe -m eval.run_eval --count 10 --mode offline

评测默认显示实时进度条，并且每完成一条用例就刷新本次带版本和时间戳的 JSONL、Markdown 与 metadata 文件。如果不需要终端进度显示，可以追加：

    .\.conda\deskpilot-py311\python.exe -m eval.run_eval --count 10 --mode offline --no-progress

默认输出文件位于 `eval/runs/` 和 `eval/reports/`，文件名包含 DeskPilot 版本、模式和测试时间；也可以用 `--output`、`--report` 和 `--metadata` 覆盖。

## 常用命令

运行当前全部 163 条：

    .\.conda\deskpilot-py311\python.exe -m eval.run_eval --count 163 --mode offline

只运行 RAG：

    .\.conda\deskpilot-py311\python.exe -m eval.run_eval --count 6 --subset RAG-DocQA --mode offline

执行网页样例。该模式会访问真实网络：

    .\.conda\deskpilot-py311\python.exe -m eval.run_eval --count 30 --subset Web-Research --mode online

指定数据集和输出位置：

    .\.conda\deskpilot-py311\python.exe -m eval.run_eval --dataset eval/dataset/deskpilot_bench.jsonl --count 5 --output eval/runs/smoke.jsonl --report eval/reports/smoke.md

评分优先采用确定性检查：路由步骤、关键事实、权限确认、证据数量和执行轨迹。第一版不把 LLM Judge 作为硬依赖。

## 评测指标

- Accuracy：通过用例数 / 实际执行用例数。
- Exact Match、Token F1：只统计带 `expected.reference_answer` 或
  `expected.reference_answers` 的事实型用例；开放式答案显示 N/A。
- Response Time：记录逐用例耗时，并汇总 Mean、P50 和 P95。
- Token Usage：读取模型服务返回的真实 `usage`；服务端不返回时显示 N/A。
- Error Rate：未捕获执行错误数 / 实际执行用例数。
- Failure Recovery：`Recovery` 子集中的通过比例。
- Task Completion：只对用例实际声明的行为断言计算完成比例。
- Communication Efficiency：当前使用任务完成度除以重试和失败惩罚的单 Agent
  代理指标，不代表多智能体通信质量。

`score` 现在等于 Task Completion；`legacy_score` 保留旧版七项固定等权分数，
用于和历史报告比较。

## Reranker 消融

本地精排器消融不会自动下载模型；`--model` 必须指向已下载的本地目录：

```powershell
D:\broagent\.conda\deskpilot-py311\python.exe -m eval.run_reranker_ablation `
  --providers rrf,lexical,cross_encoder `
  --model D:\models\mmarco-mMiniLMv2-L12-H384-v1
```

报告输出 Hit Rate、Recall、MRR、平均/P95 检索延迟、Token 和模型磁盘占用。磁盘有限时，可先用 `--providers rrf,lexical` 运行零额外模型版本。

## RAGAS 文档 RAG 评测

在项目环境中安装可选依赖：`python -m pip install -r requirements-ragas.txt`。需要 `.env` 中可用的 LLM 和 Embedding API Key；脚本强制关闭本地 fallback，模型/网络失败不会生成伪分。小样本与完整集命令：

```powershell
.\.conda\deskpilot-py311\python.exe -m eval.run_ragas_eval --count 2
.\.conda\deskpilot-py311\python.exe -m eval.run_ragas_eval
```

默认使用 `eval/dataset/ragas_cases.jsonl` 的 8 条人工编写并核对参考答案的合成故障任务。每条包含 `id/subset/query/reference/corpus/top_k/relevant_sources`；`corpus` 必须是仓库内已存在的相对路径。`--count N` 取前 N 条，`--dataset` 可换扩展集，`--report` 可指定报告路径。另有 `--prepare-only` 只生成真实检索和回答样本，不产生 RAGAS 分数。

本机 `多模态/` 下 4 篇真实论文对应的 24 条扩展用例见 [`README_ragas.md`](dataset/README_ragas.md)。PDF 原文未提交到仓库，其他机器没有这些文件时不要运行该集合。运行本地论文集前三条，或合并两个集合共 32 条：

```powershell
.\.conda\deskpilot-py311\python.exe -m eval.run_ragas_eval --dataset eval/dataset/ragas_multimodal_local.jsonl --count 3 --report eval/reports/ragas_papers_smoke.md
.\.conda\deskpilot-py311\python.exe -m eval.run_ragas_eval --dataset eval/dataset/ragas_cases.jsonl --dataset eval/dataset/ragas_multimodal_local.jsonl --report eval/reports/ragas_combined.md
```

同一组 `corpus` 在一次运行中复用索引，减少 PDF 重复解析与 Embedding 请求；不同语料组仍分别建索引。`--dataset` 可重复指定，`--count` 在合并后取前 N 条。

流程为逐例隔离建索引、调用 DeskPilot 的混合检索和 lexical 精排、用项目 LLM 据证据生成回答，随后 RAGAS 0.2.15 基于问题、实际检索片段、回答与参考答案计算 `faithfulness`（回答事实受证据支持）、`answer_relevancy`（回答针对问题）、`context_precision`（相关片段排名）和 `context_recall`（证据覆盖参考答案）。另计来源 Recall 作为确定性辅助指标；不把它冒充 RAGAS 分数。输出 `eval/reports/ragas_latest.md` 和同名 `.jsonl`，逐例保留上下文、答案、trace、错误和耗时。任何缺失/NaN 指标都显示 `N/A`，有效样本数单列。报告含原文，私人语料请使用受控输出目录。

这是隔离的文档 RAG 管线评测，不含桌面 UI、意图路由、多 Agent、工具副作用，也不是 RAGAS 官方公开榜单分数；小样本、合成语料及 LLM Judge 波动意味着只能用于本项目回归。RAGAS/LangChain 仅用于评测侧，未接入主产品实现。

## Jev 决策与 Judge 对照

可选 Jev 通过 OpenRouter `alpha.decisions.create` 对应的 `/api/alpha/decisions` 端点调用 `typesafe/jev-1.13`。入口仅为路由前置门控、普通文档 RAG 候选证据门控；Judge 是独立评测脚本，不参与线上最终回答或网页检索流程。`JEV_MODE=off/shadow/gated`，本地文档正文默认禁止外发；配置和数据构成见 [Jev 数据说明](dataset/README_jev.md)。

`JEV_PROVIDER=netlify` 时，Jev 评测入口会使用 `NETLIFY_JEV_ENDPOINT` 和 `NETLIFY_JEV_TOKEN` 调用自建 Netlify Function；`openrouter` 仍为默认值。先验证 Function 的 typed answers 契约与鉴权，再运行付费评测。该桥接的 SDK/网关行为尚未在真实站点验证，不能把模拟传输测试当作模型质量或延迟对照。

```powershell
python -m eval.compare_jev
python -m eval.run_jev_judge --judge both --count 8
```

前者在同一 36 条真实 API 套件上跑 off/gated 各一遍，后者在 8 条标注证据上计算相关性/忠实度 P/R/F1；二者都需要相应 API Key，会产生付费请求。本地没有 Key 时只运行单测，不制造“真实 API 实验数据”。

P2 后默认专项数据集为 68 条（原 P0/P1 60 条 + P2 8 条）。执行 `python -m eval.run_reliability_eval --count 68`；这是临时文件与模拟邮箱的故障注入，不含真实邮件投递。下方 60 条说明作为旧版基线范围保留。

```powershell
.\.conda\deskpilot-py311\python.exe -m eval.run_reliability_eval --count 60
.\.conda\deskpilot-py311\python.exe -m eval.run_reliability_eval --scenario checkpoint_process_restart --count 5
```

独立数据集 `dataset/reliability_cases.jsonl` 为 12 类故障各 5 次确定性实验，不调用真实模型或邮箱，不混入 DeskPilotBench 的答案 Accuracy/F1。报告位于 `reports/reliability/latest.md/latest.json`，包含真实进程退出、审批重建、附件指纹、取消和有界重试的观测范围。格式和扩展方法见 [可靠性数据说明](dataset/README_reliability.md)。

`python -m eval.prepare_reliability_demo` 可生成无隐私的资料、附件和长 PDF，用于桌面任务面板手工验证；保留已有同名文件，不覆盖。

## Jev 对照

`python -m eval.compare_jev` 对固定 36 条 API 用例做 off/gated 成对运行；报告同时给出 Jev 实际跳过原 Router 的次数与 RAG 证据过滤数，零干预时不可解读为有效模型对照。`python -m eval.run_jev_judge --judge both --count 8` 对标注证据做五档数值相关性、忠实度评分，结果为 0–1，报告 MAE/RMSE，非正式 RAGAS 指标。详见 [Jev 数据说明](dataset/README_jev.md)。

## P2 私有 API 与版本轮换

`dataset/p2_api_cases.jsonl` 含 12 条无外部副作用的 HTTP 合约用例：原有 8 条覆盖幂等重放、跨主体任务隔离、会话延续、产物、Token、SSE 和配额；新增 4 条覆盖沙箱默认关闭、独立 `execute` scope、输入路径穿越拒绝和沙箱任务幂等。它们由 `eval.run_p2_api_eval` 使用内存 ASGI 客户端和隔离 SQLite 运行，不调用真实 LLM/邮箱或 Docker。并发竞态、审批重放/附件篡改、过期 Token 等逆向用例在 `tests/test_p2_api_adversarial.py`；Docker 参数和真实越权用例在 `tests/test_p2_docker_sandbox.py`。数据集通过不代表 OS 沙箱通过。

```powershell
.\.conda\deskpilot-py311\python.exe -m eval.run_p2_api_eval --mode offline --project-version 0.9.0-beta.2
.\.conda\deskpilot-py311\python.exe -m pytest -q tests/test_p2_api_contract.py tests/test_p2_api_adversarial.py
.\.conda\deskpilot-py311\python.exe -m eval.release_validation --artifact D:\broagent\dist\NEW_BUILD\DeskPilot\DeskPilot.exe --run-offline
```

固定用例用于跨版本同题比较；`release_validation` 按上轮 manifest 优先换题，并在报告中记录所选 ID、数据 SHA、EXE 哈希与是否重复。它运行的是源码评测，打包后的 UI/API/worker 仍须用独立 profile 单独冒烟。真实文本 API/VLM 项默认 `not_run`，不能并入离线通过率；私有 API 尚未通过容器/VM 越权测试，不对公网开放。

真实 Docker 越权测试须显式设 `DESKPILOT_RUN_DOCKER_SECURITY_TESTS=1` 并准备本地 Python 镜像；启用后 daemon 或镜像不可用会**失败**，不算跳过。命令、威胁边界与固定镜像方式见 [私有 API 说明](../deskpilot/api/README.md)。
