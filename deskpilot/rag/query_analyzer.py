from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from typing import Callable

from ..core.json_utils import parse_json_value


@dataclass(frozen=True)
class QueryAnalysis:
    standalone_question: str
    entities: tuple[str, ...] = ()
    must_terms: tuple[str, ...] = ()
    task_type: str = "fact"
    needs_multi_query: bool = False
    needs_hyde: bool = False
    sub_questions: tuple[str, ...] = ()
    context_expansion: str = "none"
    method: str = "deterministic"
    cache_hit: bool = False


class QueryAnalyzer:
    """复杂查询使用结构化 LLM 分析，简单查询保持零额外模型调用。"""

    def __init__(self, llm_call: Callable[[str], str] | None = None) -> None:
        self.llm_call = llm_call
        self._cache: dict[str, QueryAnalysis] = {}

    def analyze(
        self,
        question: str,
        recent_context: str = "",
        index_version: str = "",
        complex_task: bool = False,
    ) -> QueryAnalysis:
        key = hashlib.sha256(
            f"{question}\n{recent_context[:1200]}\n{index_version}\n{complex_task}".encode("utf-8")
        ).hexdigest()
        if key in self._cache:
            value = self._cache[key]
            return QueryAnalysis(**{**value.__dict__, "cache_hit": True})

        result = self._deterministic(question, complex_task)
        if complex_task and self.llm_call and os.getenv("RAG_MULTI_QUERY_ENABLED", "true").lower() in {
            "1", "true", "yes", "on",
        }:
            parsed = self._with_llm(question, recent_context)
            if parsed is not None:
                result = parsed
        self._cache[key] = result
        return result

    def generate_hyde(self, analysis: QueryAnalysis) -> str | None:
        enabled = os.getenv("RAG_HYDE_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
        if not enabled or not analysis.needs_hyde or not self.llm_call:
            return None
        response = self.llm_call(
            "请写一个不超过120字的假设性相关文档片段，只用于向量检索。"
            "不得写引用、来源或确定性答案。问题：" + analysis.standalone_question
        ).strip()
        return response[:500] or None

    def _with_llm(self, question: str, recent_context: str) -> QueryAnalysis | None:
        prompt = (
            "你是 RAG Query Analyzer，只输出 JSON，不回答问题。最多生成3个互补检索子问题；"
            "原问题会被系统始终保留，所以不要重复原句。只有抽象描述且词项不足时 needs_hyde=true。\n"
            "schema={standalone_question:string,entities:string[],must_terms:string[],"
            "task_type:fact|summary|comparison|procedure|citation,needs_multi_query:boolean,"
            "needs_hyde:boolean,sub_questions:string[],context_expansion:sentence_window|parent|none}\n"
            f"最近必要上下文：{recent_context[-1200:]}\n当前问题：{question}"
        )
        response = self.llm_call(prompt)
        data = parse_json_value(response or "", dict)
        if not isinstance(data, dict):
            return None
        standalone = str(data.get("standalone_question") or question).strip()[:500]
        sub_questions = tuple(
            str(value).strip()[:300] for value in data.get("sub_questions", [])
            if str(value).strip() and str(value).strip() != standalone
        )[:3]
        task_type = str(data.get("task_type", "fact"))
        if task_type not in {"fact", "summary", "comparison", "procedure", "citation"}:
            task_type = "fact"
        expansion = str(data.get("context_expansion", "none"))
        if expansion not in {"sentence_window", "parent", "none"}:
            expansion = "none"
        return QueryAnalysis(
            standalone_question=standalone,
            entities=tuple(str(value)[:100] for value in data.get("entities", []) if str(value).strip())[:12],
            must_terms=tuple(str(value)[:100] for value in data.get("must_terms", []) if str(value).strip())[:12],
            task_type=task_type,
            needs_multi_query=bool(data.get("needs_multi_query")) and bool(sub_questions),
            needs_hyde=bool(data.get("needs_hyde")),
            sub_questions=sub_questions,
            context_expansion=expansion,
            method="llm",
        )

    @staticmethod
    def _deterministic(question: str, complex_task: bool) -> QueryAnalysis:
        entities = tuple(dict.fromkeys(re.findall(
            r"[A-Za-z][A-Za-z0-9]*(?:[-_.][A-Za-z0-9]+)*|[A-Z]{2,}[A-Za-z0-9]*|\d+(?:\.\d+)+",
            question,
        )))
        return QueryAnalysis(
            standalone_question=" ".join(question.split()),
            entities=entities[:12],
            must_terms=entities[:8],
            task_type="comparison" if complex_task else "fact",
            needs_multi_query=False,
            needs_hyde=False,
            context_expansion="parent" if complex_task else "sentence_window",
        )
