# DeskPilot Multimodal

该模块实现多模态 P0/P1/P2，并保持原图、派生文本、记忆引用和向量分离：

- `AssetStore` 将校验后的图片和缩略图放入受控资产目录，SQLite 只保存路径和元数据。
- `VisionLLMProvider` 使用 OpenAI-compatible `image_url` 消息完成单图/多图问答，并支持 SSE 增量输出。
- `PaddleOCRProvider` 与 `SigLIP2EmbeddingProvider` 都是延迟加载的可选本地 Provider。
- `MultimodalVectorStore` 按 `embedding_space + dimension` 隔离向量，禁止不同模型误算。
- `MultimodalRetriever` 用 RRF 融合 OCR 关键词、OCR 文本向量和视觉向量。
- PDF 页图在一次 OCR worker 和一次视觉 worker 调用中批量处理；worker 内批大小由 `IMAGE_EMBEDDING_BATCH_SIZE` 限制。
- OCR 文本优先复用 Embedding API 生成版本化语义向量；无 API 时降级为 `text/local-hash-v1`。
- SQLite 使用 WAL 和 busy timeout；资产删除同步清理文件、chunk 与向量。
- `VisualMemoryManager` 只在用户显式要求时创建 `pending` 视觉记忆；记忆保存文本事实、`asset_refs` 和 `evidence_refs`，不复制图片向量或 Base64。
- 审批后的视觉记忆支持跨会话文本召回和相似图片关联召回；进入上下文前同时受文本 token、图片数量和总像素预算约束。
- 剪贴板/截图、密码、Token、验证码、身份证号和银行卡号禁止进入长期视觉记忆；被记忆引用的资产删除需要人工确认和显式级联。

## 模型选择

`qwen3.8-max` 不应直接作为视觉模型配置。模型名称本身不能证明其支持图片，而且该名称需要以实际供应商模型列表为准。只有当对应部署明确支持 OpenAI-compatible `image_url` 输入时才能用于 `VISION_MODEL`。DashScope 场景应优先选择明确的 Qwen-VL 视觉型号，例如配置模板中的 `qwen-vl-max-latest`，文本路由模型仍可保持原配置。

P1 建议在本地增加 **SigLIP2 base**，因为它负责文本搜图和图搜图，是视觉索引的核心，不是生成模型。默认 `basic` fallback 只做颜色/外观近似的图搜图，不能进行可靠文本搜图。OCR 建议使用 PaddleOCR。

本地生成式 VLM 不是 P0/P1 的硬依赖。需要图片不出本机时，再单独部署 `Qwen2.5-VL-3B-Instruct` 作为较稳妥的轻量起点；显存充足可选择 7B。它与 SigLIP2 职责不同，不能替代检索向量模型。

## 安装与配置

基础图片问答：

```powershell
python -m pip install -r requirements.txt
```

本地 OCR 与 SigLIP2：

```powershell
python -m pip install -r requirements-multimodal.txt
```

至少配置：

```dotenv
VISION_MODEL=qwen-vl-max-latest
VISION_API_KEY=your_key
ALLOW_CLOUD_IMAGE_UPLOAD=true
SENSITIVE_IMAGE_POLICY=confirm
```

启用本地检索：

```dotenv
IMAGE_EMBEDDING_PROVIDER=siglip2
IMAGE_EMBEDDING_BATCH_SIZE=8
OCR_PROVIDER=paddleocr
```

`SENSITIVE_IMAGE_POLICY` 支持 `block|confirm|allow`。剪贴板/截图默认标记为敏感，默认策略会在聊天内请求单次人工确认。首次加载 SigLIP2 可能需要下载模型。DeskPilot 不会把 Base64、API Key 或原始图片写入会话 JSONL。

视觉记忆预算可选配置：

```dotenv
VISUAL_MEMORY_MAX_IMAGES=4
VISUAL_MEMORY_MAX_TOTAL_PIXELS=12000000
```

离线验证：

```powershell
python -m pytest tests\test_multimodal_memory_p2.py -q
python -m eval.run_multimodal_eval --mode offline --count 22
```
