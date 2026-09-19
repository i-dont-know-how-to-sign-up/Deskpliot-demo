# DeskPilotBench Dataset

> 主数据集现为 v0.6，共 140 条；新增 9 条 Composite-Workflow 复合操作回归用例、5 条索引感知路由回归用例、1 条 RAG 证据忠实性用例和 1 条真实双 PDF 比较用例。完整统计、运行模式与 mock 说明以 `../数据集描述文档.md` 为准。旧版 92 条统计仅为历史记录；`offline` 现禁用外部 API，不能和历史报告的同名模式直接比较。另有 3 条可选的 `local_multimodal_cases.jsonl` 本地真实 PDF 用例。

`reg_rag_compare_001` 使用本地真实 `blip.pdf` 与 `blip-2.pdf`，检查短标题、连字符变体、跨论文均衡取证、综合回答与双文档引用来源。该用例需要真实 LLM API。

`reg_rag_route_004` 在只索引含无关 `References` 标题的 BLIP fixture 后询问 Agentic RL，检查系统保持直接回答、不执行 RAG、不引用参考文献条目，也不编造 DeskPilot 的框架实现。

`reg_rag_route_005` 在索引只顺带提到 GPT-4V 的多模态综述 fixture 后询问 GPT 工作流程，检查短缩写不会因一次旁支提及触发 RAG。`local_multi_003` 使用真实 Recognize Anything PDF，检查流程问题优先召回第 3 页架构正文并生成完整引用回答。

## RAG P0 检索层数据集

`rag_p0_cases.jsonl` 是与 140 条端到端主数据集分离的检索层数据集，共 10 条。它直接评估 chunk/section 召回，避免最终 LLM 回答掩盖分块和索引问题：

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

## RAG P1 混合检索数据集

`rag_p1_cases.jsonl` 包含 8 条可扩展检索层用例，覆盖错误码、版本号、中文精确词、语义问题、跨文档 Multi-Query、metadata filter、真实 BLIP/BLIP-2 PDF 和空结果。`run_rag_p1_eval.py` 支持 `dense|bm25|hybrid|all` 消融、`offline|api` 模式、数量限制和按 ID 执行。

```powershell
.\.conda\deskpilot-py311\python.exe -m eval.run_rag_p1_eval --mode offline --strategy all
.\.conda\deskpilot-py311\python.exe -m eval.run_rag_p1_eval --mode api --strategy hybrid --include-local --case-id rag_p1_007
```

当前包含 92 条用例：DirectQA 12、RAG-DocQA 26、Intent-Routing 12、Tool-Calling 12、Web-Research 10、Memory 8、Safety-Permission 8、Recovery 4。

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
