# DeskPilotBench 评测脚本

## 快速运行

在项目根目录执行：

    .\.conda\deskpilot-py311\python.exe -m eval.run_eval --count 10 --mode offline

评测默认显示实时进度条，并且每完成一条用例就刷新 `eval/runs/latest.jsonl` 和
`eval/reports/latest.md`。如果不需要终端进度显示，可以追加：

    .\.conda\deskpilot-py311\python.exe -m eval.run_eval --count 10 --mode offline --no-progress

输出文件：

- eval/runs/latest.jsonl：每条用例的结构化结果。
- eval/reports/latest.md：Markdown 汇总报告。

## 常用命令

运行当前全部 140 条：

    .\.conda\deskpilot-py311\python.exe -m eval.run_eval --count 140 --mode offline

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
