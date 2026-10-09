from __future__ import annotations

from ..core.api_clients import OpenAICompatibleClient, local_hash_embedding
from ..core.config import ModelConfig, load_config


class MultimodalTextEmbedder:
    """为 OCR 文本提供版本化向量；无 API 时诚实降级为本地 hash。"""

    def __init__(
        self,
        client: OpenAICompatibleClient | None = None,
        config: ModelConfig | None = None,
    ) -> None:
        self.config = config or load_config()
        self.client = client or OpenAICompatibleClient(self.config)
        self.semantic = bool(self.config.embedding_api_key)
        if self.semantic:
            self.model_id = self.config.embedding_model
            self.model_revision = "api"
            self.embedding_space = f"text/{self.model_id}@api"
        else:
            self.model_id = "deskpilot/local-hash"
            self.model_revision = "1"
            self.embedding_space = "text/local-hash-v1"

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self.semantic:
            return self.client.embed(texts)
        return [local_hash_embedding(text) for text in texts]
