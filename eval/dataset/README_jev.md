# Jev 对照评测数据说明

## 两类实验

1. **Agent 端到端对照**：复用 `eval/suites/baseline_api_v0.9.0.json` 的 36 条固定真实 API 用例（12 个功能模块，每模块简单/中等/复杂各 1 条）。`eval.compare_jev` 依次以 `JEV_MODE=off` 和 `JEV_MODE=gated` 运行相同用例，比较任务验收 Accuracy、仅有参考答案样本的答案 F1、P95 耗时、总 Token、回退/退化用例。报告单列 Jev Router 调用数、实际跳过原 Router 次数与 RAG 证据过滤数；若没有实际干预，不应把两组结果当作 Jev 效果对照。原测试套件的网页和邮件副作用依旧使用隔离 fixture/mock；这不是公开网页或私人邮箱的真实交互成功率。
2. **Judge 能力对照**：`jev_judge_cases.jsonl` 是 8 条人工构造的极小示例，RAG 5 条、网页 3 条；每条有 query、evidence、answer、`expected.relevant` 和 `expected.faithful` 的 0–1 人工分数。相关性表示证据覆盖问题的程度；忠实度表示答案事实主张受该证据支持的程度，矛盾或无依据的主张降分。Jev 和原 LLM 都按相同五档 0–4 判别，再除以 4；**Jev 的 `score` 是档位概率加权的期望值，可能为 3.91 等小数；LLM 按提示输出单个整数档位**。报告 MAE、RMSE、平均分和误差不超过一档的比例，并统计延迟/usage。这是 RAGAS 风格的**代理评分**，不是 RAGAS 的逐声明 faithfulness 或 context precision 正式实现，也不代表 Agent 整体效果。

## JSONL 结构

```json
{"id":"jev_rag_001","source_type":"rag","query":"...","evidence":"...","answer":"...","expected":{"relevant":1.0,"faithful":1.0},"source":"synthetic_from_fixture:..."}
```

`id` 唯一；`source_type` 为 `rag` 或 `web`；`source` 标明来源和构造方式。后续扩容应添加真实检索输出、来源 URL/文档 ID、人工复核标签和分层划分，避免用运行中的模型预测充当 gold label。当前只有 8 条，指标方差很大，不得据此宣称模型全面优于原 LLM。

## 执行

先在项目根目录 `.env` 中配置 `OPENROUTER_API_KEY`、`DASHSCOPE_API_KEY`（或等价 LLM/Embedding 配置）。不要提交 Key。运行：

```powershell
D:\broagent\.conda\deskpilot-py311\python.exe -m eval.compare_jev
D:\broagent\.conda\deskpilot-py311\python.exe -m eval.run_jev_judge --judge both --count 8
```

首条会触发 **72 次 Agent 用例执行**（36 条 x 两组）和相应真实 LLM/Embedding/Jev 请求，耗时与费用取决于服务端。输出在 `eval/runs/jev_compare_<时间>/comparison.md`；第二条输出在 `eval/reports/jev_judge/latest.json`。比较现有两份 36 条运行结果可用 `eval.compare_jev --off-run ... --gated-run ...`，此模式不重新调用 API。必须确保两组用例 ID 相同；若 Jev 成功调用数为零，不能把全回退结果解读为 Jev 的质量。

Jev 只接入路由前置门控和普通文档 RAG 初排后的证据门控；Judge 是独立评测脚本，不改变生产网页检索、最终回答或记忆链路。默认关闭；`shadow` 只观察；`gated` 可在高置信低风险场景改变行为。RAG 文本默认不外发，需显式设置 `JEV_ALLOW_DOCUMENT_TEXT=true` 才测试 Jev RAG 门控。
