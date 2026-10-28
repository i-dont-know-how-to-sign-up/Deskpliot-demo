# RAGAS 小型评测集

`ragas_cases.jsonl` 当前 8 条，全部依据仓库内的 NEBULA 客户端、Aurora Cache 和 Falcon Scheduler 合成故障文档人工编写参考答案；不是 RAGAS/GAIA 官方数据。覆盖 procedure 3、contrast 1、causal 1、architecture 1、multi_document 1、limitation 1。每例混入至少一份非目标文档，避免只有单文档命中的虚高评分。

每行 JSON 对象示例：

```json
{"id":"ragas_nebula_712","subset":"procedure","query":"NEBULA-712 出现后客户端应如何恢复？","reference":"...","corpus":["eval/dataset/fixtures/docs/rag_p2_nebula.md"],"top_k":4,"relevant_sources":["rag_p2_nebula.md"]}
```

`query` 是实际检索/生成输入；`reference` 为独立于模型输出的人工标准答案；`corpus` 为允许索引的仓库内相对路径；`relevant_sources` 仅用于来源 Recall，不直接传给 RAGAS Judge。运行时产生的 `retrieved_contexts` 和 `response` 才传入 RAGAS，不得用标准答案代替模型输出。扩展时按来源/任务难度分层，并保留未参与提示或调参的测试集；增加人工复核记录和真实脱敏文档后才适合判断泛化效果。

## 本地真实论文扩展集

`ragas_multimodal_local.jsonl` 增加 24 条人工构造的问答：BLIP、BLIP-2、Recognize Anything、MLLM Survey 各 5 条单论文题，另有 4 条跨论文比较。每题记录 `gold_pages`（PDF 文件中的 1-based 页号）和 `evidence_anchor`（可在这些页提取文本中核查的原文锚点）。页码/锚点仅用于人工审计和本地测试，不发送给 RAGAS Judge。PDF 本体位于 `多模态/`，未上传仓库；CI 没有这些私有资产时跳过 PDF 锚点测试。文档是实际论文，但**问题与参考答案是本项目人工编写的，不是论文作者或公开 QA 数据集提供的标准题**。

扩展集合包含较长 PDF，RAGAS 评分会产生较多 Embedding、答案生成和 Judge API 调用，建议先 `--count 2` 冒烟，再运行全量。`--dataset` 可重复指定以合并默认 8 条和本地 24 条；完整命令见 `eval/README.md`。注意报告 JSONL 保留检索正文，不要在使用私人文档时公开。
