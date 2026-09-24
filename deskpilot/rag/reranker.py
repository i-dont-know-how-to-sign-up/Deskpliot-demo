from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import Protocol

from ..core.api_clients import cosine_similarity, local_hash_embedding


@dataclass(frozen=True)
class RerankDocument:
    """精排只接收有界 child 文本和结构元数据，不接收完整 parent。"""

    chunk_id: str
    text: str
    title: str = ""
    heading_path: tuple[str, ...] = ()
    page_number: int | None = None
    base_score: float = 0.0

    def model_text(self, max_chars: int = 2400) -> str:
        heading = " / ".join(self.heading_path)
        prefix = "\n".join(
            value for value in (
                f"文件：{self.title}" if self.title else "",
                f"标题：{heading}" if heading else "",
                f"页码：{self.page_number}" if self.page_number else "",
            ) if value
        )
        return f"{prefix}\n正文：{self.text[:max_chars]}".strip()


@dataclass(frozen=True)
class RerankScore:
    chunk_id: str
    score: float


class Reranker(Protocol):
    provider: str

    def rerank(self, query: str, documents: list[RerankDocument]) -> list[RerankScore]: ...


def _tokens(text: str) -> list[str]:
    lowered = text.casefold()
    values = re.findall(r"[a-z][a-z0-9_.+-]{1,}|[\u4e00-\u9fff]{2,}", lowered)
    for group in re.findall(r"[\u4e00-\u9fff]{3,}", lowered):
        values.extend(group[index:index + 2] for index in range(len(group) - 1))
    return list(dict.fromkeys(values))


class DisabledReranker:
    provider = "disabled"

    def rerank(self, query: str, documents: list[RerankDocument]) -> list[RerankScore]:
        return [RerankScore(item.chunk_id, item.base_score) for item in documents]


class LexicalCrossEncoderReranker:
    """零依赖的 query-document 成对打分 fallback。

    它不是神经 Cross-Encoder，但使用与 Cross-Encoder 相同的成对输入边界，便于
    未安装模型时保持 P2 流水线可运行，并在 trace 中明确标记 lexical。
    """

    provider = "lexical"

    def rerank(self, query: str, documents: list[RerankDocument]) -> list[RerankScore]:
        query_tokens = _tokens(query)
        query_phrase = " ".join(re.findall(r"[a-z0-9_.+-]+", query.casefold()))
        results: list[RerankScore] = []
        for item in documents:
            body = item.model_text().casefold()
            matched = sum(self._contains(token, body) for token in query_tokens)
            coverage = matched / max(1, len(query_tokens))
            phrase_bonus = 0.12 if query_phrase and len(query_phrase) >= 6 and query_phrase in body else 0.0
            title_tokens = _tokens(item.title + " " + " ".join(item.heading_path))
            title_overlap = sum(token in query_tokens for token in title_tokens) / max(1, len(query_tokens))
            score = min(1.0, coverage * 0.72 + title_overlap * 0.16 + phrase_bonus)
            results.append(RerankScore(item.chunk_id, score))
        return sorted(results, key=lambda value: value.score, reverse=True)

    @staticmethod
    def _contains(token: str, text: str) -> bool:
        if re.fullmatch(r"[a-z0-9_.+-]+", token):
            return bool(re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", text))
        return token in text


class SentenceTransformersCrossEncoderReranker:
    provider = "cross_encoder"

    def __init__(self, model_name: str) -> None:
        if not model_name.strip():
            raise RuntimeError("RAG_RERANK_MODEL 未配置，拒绝隐式下载 Cross-Encoder 模型")
        try:
            from sentence_transformers import CrossEncoder  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("未安装 sentence-transformers") from exc
        # local_files_only 防止桌面问答时意外下载数 GB 模型。
        self.model = CrossEncoder(model_name, automodel_args={"local_files_only": True})

    def rerank(self, query: str, documents: list[RerankDocument]) -> list[RerankScore]:
        if not documents:
            return []
        scores = self.model.predict([(query, item.model_text()) for item in documents])
        values = [float(value) for value in scores]
        # 不同模型可能输出 logit 或 0-1 分数，统一映射到 0-1。
        normalized = [value if 0.0 <= value <= 1.0 else 1.0 / (1.0 + math.exp(-value)) for value in values]
        return sorted(
            (RerankScore(item.chunk_id, score) for item, score in zip(documents, normalized)),
            key=lambda value: value.score,
            reverse=True,
        )


class HashColBERTReranker:
    """实验性 late-interaction provider，用本地 token hash 向量验证 ColBERT 接口。

    该实现不冒充训练版 ColBERT，也不创建大型 token 索引；适合离线消融和接口
    验证。生产质量需替换为带 revision/namespace 的真实 ColBERT provider。
    """

    provider = "colbert_experimental_hash"

    def rerank(self, query: str, documents: list[RerankDocument]) -> list[RerankScore]:
        query_tokens = _tokens(query)[:32]
        query_vectors = [local_hash_embedding(token) for token in query_tokens]
        results: list[RerankScore] = []
        for item in documents:
            document_tokens = _tokens(item.model_text())[:256]
            document_vectors = [local_hash_embedding(token) for token in document_tokens]
            if not query_vectors or not document_vectors:
                score = 0.0
            else:
                maxima = [max(cosine_similarity(query_vector, vector) for vector in document_vectors)
                          for query_vector in query_vectors]
                score = max(0.0, min(1.0, sum(maxima) / len(maxima)))
            results.append(RerankScore(item.chunk_id, score))
        return sorted(results, key=lambda value: value.score, reverse=True)


class ApiReranker:
    provider = "api"

    def __init__(self, endpoint: str, api_key: str = "", model: str = "") -> None:
        if not endpoint.strip():
            raise RuntimeError("RAG_RERANK_API_URL 未配置")
        self.endpoint = endpoint
        self.api_key = api_key
        self.model = model

    def rerank(self, query: str, documents: list[RerankDocument]) -> list[RerankScore]:
        import requests

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = requests.post(
            self.endpoint,
            headers=headers,
            json={
                "model": self.model,
                "query": query,
                "documents": [item.model_text() for item in documents],
                "top_n": len(documents),
            },
            timeout=max(3, min(int(os.getenv("RAG_RERANK_TIMEOUT_SECONDS", "20")), 120)),
        )
        response.raise_for_status()
        data = response.json()
        raw_results = data.get("results", data.get("data", []))
        scores: list[RerankScore] = []
        for result in raw_results:
            index = int(result.get("index", -1))
            if 0 <= index < len(documents):
                score = float(result.get("relevance_score", result.get("score", 0.0)))
                scores.append(RerankScore(documents[index].chunk_id, score))
        if not scores:
            raise RuntimeError("Rerank API 未返回可解析结果")
        return sorted(scores, key=lambda value: value.score, reverse=True)


def build_reranker(provider: str | None = None) -> Reranker:
    selected = (provider or os.getenv("RAG_RERANK_PROVIDER", "lexical")).strip().lower()
    if selected in {"", "disabled", "none", "rrf"}:
        return DisabledReranker()
    if selected == "lexical":
        return LexicalCrossEncoderReranker()
    if selected == "cross_encoder":
        return SentenceTransformersCrossEncoderReranker(os.getenv("RAG_RERANK_MODEL", ""))
    if selected == "colbert":
        return HashColBERTReranker()
    if selected == "api":
        return ApiReranker(
            os.getenv("RAG_RERANK_API_URL", ""),
            os.getenv("RAG_RERANK_API_KEY", ""),
            os.getenv("RAG_RERANK_MODEL", ""),
        )
    raise ValueError(f"不支持的 RAG_RERANK_PROVIDER：{selected}")
