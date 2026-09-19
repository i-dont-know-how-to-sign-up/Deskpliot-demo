from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class RewrittenQuery:
    original: str
    rewritten: str
    method: str


class QueryOptimizer:
    """为检索去除动作噪声；可选 LLM 改写失败时保持确定性降级。"""

    # 本地 hash embedding 不具备跨语言语义能力。这里只扩展通用检索方面，
    # 不根据具体产品、论文名或业务动作决定路由。
    MULTILINGUAL_ASPECTS = (
        (("工作流程", "处理流程", "流程"), "workflow architecture process pipeline"),
        (("架构", "结构"), "architecture structure modules"),
        (("原理", "机制"), "principle mechanism method"),
        (("训练", "训练过程"), "training objective procedure"),
        (("推理", "推理过程"), "inference procedure"),
        (("区别", "差异", "对比", "比较"),
         "comparison differences similarities architecture components training objectives inference efficiency"),
    )

    def __init__(self, llm_call: Callable[[str], str] | None = None) -> None:
        self.llm_call = llm_call

    def rewrite(self, query: str, context: str = "", use_llm: bool = False) -> RewrittenQuery:
        original = " ".join(query.strip().split())
        if use_llm and self.llm_call:
            response = self.llm_call(
                "把以下请求改写为一条用于本地文档向量检索的短查询，只输出查询，不回答问题：\n"
                f"上下文：{context}\n请求：{original}"
            ).strip()
            if response and len(response) <= 300:
                return RewrittenQuery(original, response, "llm")
        rewritten = re.sub(
            r"^(请|麻烦|帮我|请帮我)?\s*(读取|打开|查看|根据|基于|搜索|检索|总结|分析|回答)\s*",
            "",
            original,
        )
        rewritten = re.sub(r"(请|并且|然后)?\s*(给出|生成)?\s*(详细)?(回答|总结|说明)[。.!！]?$", "", rewritten)
        rewritten = rewritten.strip(" ，,。.!！？?") or original
        expanded = self._expand_multilingual_aspects(rewritten)
        if expanded != rewritten:
            return RewrittenQuery(original, expanded, "deterministic_multilingual")
        return RewrittenQuery(original, rewritten, "deterministic" if rewritten != original else "identity")

    def _expand_multilingual_aspects(self, query: str) -> str:
        """为中英混合技术查询补充英文方面词，改善英文文档的离线召回。"""
        if not re.search(r"[A-Za-z][A-Za-z0-9+_.-]*", query) or not re.search(r"[\u4e00-\u9fff]", query):
            return query
        expansions: list[str] = []
        for aliases, terms in self.MULTILINGUAL_ASPECTS:
            if any(alias in query for alias in aliases):
                expansions.append(terms)
        return f"{query} {' '.join(expansions)}".strip() if expansions else query
