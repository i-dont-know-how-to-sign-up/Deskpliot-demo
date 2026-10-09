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
