# DeskPilotBench Dataset

> 主数据集现为 v0.8，共 152 条；除既有复合操作、索引感知路由和 RAG 回归用例外，新增 Supervisor 真实执行链、Memory Gate 和 `debug3.md` 历史问题看护用例。完整统计、运行模式与 mock 说明以 `../数据集描述文档.md` 为准。旧版 92/140/146 条统计仅为历史记录；`offline` 现禁用外部 API，不能和历史报告的同名模式直接比较。另有 3 条可选的 `local_multimodal_cases.jsonl` 本地真实 PDF 用例。

`reg_rag_compare_001` 使用本地真实 `blip.pdf` 与 `blip-2.pdf`，检查短标题、连字符变体、跨论文均衡取证、综合回答与双文档引用来源。该用例需要真实 LLM API。

`reg_rag_route_004` 在只索引含无关 `References` 标题的 BLIP fixture 后询问 Agentic RL，检查系统保持直接回答、不执行 RAG、不引用参考文献条目，也不编造 DeskPilot 的框架实现。

`reg_rag_route_005` 在索引只顺带提到 GPT-4V 的多模态综述 fixture 后询问 GPT 工作流程，检查短缩写不会因一次旁支提及触发 RAG。`local_multi_003` 使用真实 Recognize Anything PDF，检查流程问题优先召回第 3 页架构正文并生成完整引用回答。

## RAG P0 检索层数据集

`rag_p0_cases.jsonl` 是与 152 条端到端主数据集分离的检索层数据集，共 10 条。它直接评估 chunk/section 召回，避免最终 LLM 回答掩盖分块和索引问题：

- Markdown 标题、列表和表格结构：3 条。
- TXT 语义主题边界：1 条。
- 页面边界：1 条。
- `多模态` 目录中的 Markdown/CSV 本地 fixture：2 条。
- 跨文档召回：1 条。
- BLIP 与 BLIP-2 真实 PDF 可选长文测试：2 条。

模板：

```json
{
  "id": "rag_p0_001",
  "subset": "structure_markdown",
  "source": "synthetic|local_fixture|local_long_document",
  "corpus": ["eval/dataset/fixtures/docs/example.md"],
  "query": "检索问题",
  "relevant_sources": ["example.md"],
  "relevant_contains": ["gold evidence keyword"],
  "top_k": 5,
  "runtime": {"optional_local": false}
}
```

默认运行 8 条轻量用例；增加 `--include-local` 后索引两篇真实 PDF：

```powershell
.\.conda\deskpilot-py311\python.exe -m eval.run_rag_p0_eval --count 8
.\.conda\deskpilot-py311\python.exe -m eval.run_rag_p0_eval --include-local
```

输出包括 Recall、MRR、nDCG、延迟、选中来源和完整 `RetrievalTrace`。该数据集可直接追加 JSONL；新增语料只能使用仓库内相对路径，runner 会拒绝缺失文件和越出项目根目录的路径。

当前版本：v0.1-p0

## RAG P2 精排与上下文扩展数据集

`rag_p2_cases.jsonl` 包含 12 条可移植合成用例，使用 NEBULA、Aurora Cache 和 Falcon Scheduler 四份 fixture，覆盖 Sentence Window、标题边界、Parent expansion、精排、相关性精度、错误码、跨文档覆盖、MMR、上下文预算和 provider fallback。该集合刻意不复用此前 Demo 中的 BLIP/BLIP-2、Q-Former、RAM、GPT、Agentic RL 或 ATLAS-503 问题。

字段在通用 `corpus/query/top_k` 基础上增加：`expansion`、`provider`、`max_context_tokens`、`required_terms` 和 `forbidden_terms`。runner 输出来源 Recall、事实项 Recall、Precision@K、MRR、nDCG、上下文 token、证据冗余度、延迟、effective provider 和完整 trace。

```powershell
.\.conda\deskpilot-py311\python.exe -m eval.run_rag_p2_eval --count 12 --mode offline --provider auto --expansion auto
.\.conda\deskpilot-py311\python.exe -m eval.run_rag_p2_eval --case-id rag_p2_007 --provider all --expansion all
```

`cross_encoder` 与 `api` 不包含在默认 `all` 中，避免隐式模型下载或真实计费；需要单独显式选择并完成配置。

### P2 Demo 手工验证用例

1. 导入 `rag_p2_nebula.md`，提问：“NEBULA-712 的完整恢复流程是什么？为什么不能直接重试原请求？”应看到 `context_expansion=sentence_window`，回答包含刷新租约、checkpoint、校验读和旧主节点风险；首条 Evidence 不能混入 `NEBULA-408` 章节。
2. 导入 `rag_p2_aurora.md`，提问：“总结 Aurora Cache 的一致性方案、节点故障恢复和已知限制，并分别给出依据。”应看到 `context_expansion=parent`，三个主题均有来源；Evidence token 不超过配置预算，同一 parent 不重复出现多个相同窗口。
3. 同时导入 `rag_p2_falcon_fairness.md` 与 `rag_p2_falcon_recovery.md`，提问：“比较 Falcon Scheduler 的公平性策略和过载恢复策略，并分析二者的资源权衡。”应在最终 Evidence 中保留两份文档，Steps 的 `select_evidence` 显示 `coverage_aware_mmr`，回答同时包含 deficit/aging 与高低水位/延迟队列。

三组用例都应在 Steps 中出现 `rerank_evidence`、`expand_evidence`、`select_evidence` 和 `retrieval_trace`。默认 provider 应显示 `effective=lexical`；将 `RAG_RERANK_PROVIDER` 配成不可用 provider 时，应显示 fallback 而不是整轮问答失败。

## RAG P1 混合检索数据集

`rag_p1_cases.jsonl` 包含 8 条可扩展检索层用例，覆盖错误码、版本号、中文精确词、语义问题、跨文档 Multi-Query、metadata filter、真实 BLIP/BLIP-2 PDF 和空结果。`run_rag_p1_eval.py` 支持 `dense|bm25|hybrid|all` 消融、`offline|api` 模式、数量限制和按 ID 执行。

```powershell
.\.conda\deskpilot-py311\python.exe -m eval.run_rag_p1_eval --mode offline --strategy all
.\.conda\deskpilot-py311\python.exe -m eval.run_rag_p1_eval --mode api --strategy hybrid --include-local --case-id rag_p1_007
```

主数据集当前包含 152 条用例。`safe_009` 至 `safe_011` 用于看护 LLM 伪造 `confirm`、危险 Python 静态阻断和受保护目录移动；`multi_exec_001` 用于看护 Planner、PlanExecutor、Supervisor 与人工确认的完整 DAG；`memory_gate_001` 至 `memory_gate_002` 用于看护低价值问答跳过记忆抽取和显式偏好进入记忆处理；`reg_debug3_001` 至 `reg_debug3_006` 覆盖只读 PowerShell 统计、短文件原文、已有附件邮件、显式知识库、RAM 缩写别名和冲突输出路径。具体子集数量以 `deskpilot_bench.jsonl` 实时统计为准。

GAIA/Ragas 改编来源清单位于 dataset/public_sources/gaia_ragas_adaptation_manifest.json。

邮件 MCP 测试用例位于 `email_mcp_cases.jsonl`，共 6 条；使用 `EMAIL_PROVIDER=mock`

## 多智能体 P0 用例

`deskpilot_bench.jsonl` 中新增 `multi_001` 至 `multi_005`，共 5 条，覆盖：

- Planner 先行和顺序计划；
- BFCL 风格的多工具依赖与人工确认约束；
- GAIA 风格的跨资料、多步骤任务；
- 网页/文档资料到邮件或报告的跨 Agent 协作；
- 高质量任务的 Reflection 启用条件。

这些用例是基于 BFCL/GAIA 任务形式设计的公开数据集风格样例，不是原始数据集的逐题复制。原始 GAIA 的完整测试集通常需要按其许可和评测协议获取，当前评测仍使用本地可复现数据。
即可离线验证读取、搜索、分类、线程摘要和发送/保存草稿的权限审批。

主数据文件：dataset/deskpilot_bench.jsonl。每行是一个独立 JSON 对象，新增用例时只需追加一行。

主 runner 支持重复传入 `--case-id` 精确执行回归用例，例如：

```powershell
.\.conda\deskpilot-py311\python.exe -m eval.run_eval --case-id reg_rag_route_001 --mode offline
```

## 用例模板

{
  "id": "unique_case_id",
  "subset": "RAG-DocQA",
  "source": "manual|GAIA-adapted|BFCL-adapted|regression",
  "input": "single-turn user input",
  "conversation": ["message 1", "message 2"],
  "setup": {"index_files": ["relative/path.md"]},
  "expected": {
    "mode": "direct_answer|tool_call|clarify",
    "must_have_step": "step name",
    "must_include": ["required fact"],
    "must_not_include": ["forbidden claim"],
    "requires_confirmation": false
  },
  "runtime": {"requires_online": false}
}

字段说明：

- id：全局唯一 ID，建议使用模块_编号。
- subset：能力子集，可通过 runner 的 --subset 筛选。
- source：人工设计、公开基准改编或历史回归来源。
- input：单轮输入；conversation 存在时优先使用多轮输入。
- setup.index_files：运行前导入的本地 fixture。
- expected：可自动检查的期望行为，不要求回答逐字匹配。
- runtime.requires_online：需要真实网络时设为 true。

第一版不纳入 WebArena。WebArena 需要自托管网页环境，磁盘和环境成本较高；网页样例先使用 online 标记，后续可以改成固定 HTML 和 fake search。

## 多智能体 P1 用例

新增 `multi_p1_001` 至 `multi_p1_003`，覆盖：

- 独立知识节点的并行执行；
- 临时失败后的有限重试和最终状态记录；
- 高质量任务启用 Reflection、普通即时问答跳过 Reflection 的策略。

P1 用例沿用同一 JSONL 模板，新增字段可放在 `expected` 中，例如 `parallelizable`、`recovery_required` 和 `reflection_policy`。评测 runner 未识别字段时应忽略它们，不影响已有用例。
