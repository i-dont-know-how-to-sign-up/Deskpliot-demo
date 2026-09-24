from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from deskpilot.core.agent import DocumentQAAgent
from deskpilot.core.api_clients import local_hash_embedding
from deskpilot.core.models import Document
from deskpilot.intent import IntentRouter, SlotValidator
from deskpilot.rag.vector_index import DocumentIndex
from deskpilot.tools.tool_registry import ToolParameter, ToolRegistry, ToolSpec


def force_local_fallback() -> None:
    import os

    os.environ.pop("DASHSCOPE_API_KEY", None)
    os.environ.pop("LLM_API_KEY", None)
    os.environ.pop("EMBEDDING_API_KEY", None)
    os.environ["ALLOW_LOCAL_FALLBACK"] = "true"


class FakeClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.last_error = ""
        self.config = type("Config", (), {"llm_api_key": "fake"})()

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.2) -> str:
        self.last_error = ""
        return self.response


def test_slot_validator_accepts_tool_dict_and_coerces_values() -> None:
    validator = SlotValidator()
    tool = {
        "name": "files.write_file",
        "parameters": [
            {"name": "path", "type": "string", "required": True},
            {"name": "content", "type": "string", "required": True},
            {"name": "overwrite", "type": "boolean", "required": False, "default": False},
        ],
    }

    result = validator.validate(tool, {"path": "D:\\tmp\\note.txt", "content": "hello", "overwrite": "yes"})

    assert result.ok is True
    assert result.arguments["path"] == "D:\\tmp\\note.txt"
    assert result.arguments["content"] == "hello"
    assert result.arguments["overwrite"] is True


def test_slot_validator_reports_missing_required_slots() -> None:
    validator = SlotValidator()
    tool = {
        "name": "files.write_file",
        "parameters": [
            {"name": "path", "type": "string", "required": True},
            {"name": "content", "type": "string", "required": True},
        ],
    }

    result = validator.validate(tool, {"path": "D:\\tmp\\note.txt"})

    assert result.ok is False
    assert "content" in result.missing_slots


def test_intent_router_prefers_direct_answer() -> None:
    router = IntentRouter(
        FakeClient('{"mode":"direct_answer","reason":"通用问题","confidence":0.91,"arguments":{},"missing_slots":[]}')
    )

    decision = router.route("什么是 RAG", [])

    assert decision.mode == "direct_answer"
    assert "通用问题" in decision.reason


def test_offline_fallback_routes_structural_file_write() -> None:
    decision = IntentRouter(FakeClient("")).route(
        "在当前目录创建 note.txt",
        [{"name": "files.write_file", "parameters": []}],
    )
    assert decision.mode == "tool_call"
    assert decision.tool_name == "files.write_file"


def test_offline_fallback_routes_structural_python_execution() -> None:
    decision = IntentRouter(FakeClient("")).route(
        "运行 tests/test_sample.py",
        [{"name": "shell.execute_command", "parameters": []}],
    )
    assert decision.mode == "tool_call"
    assert decision.tool_name == "shell.execute_command"


def test_index_hint_is_injected_but_does_not_force_generic_question() -> None:
    router = IntentRouter(
        FakeClient('{"mode":"direct_answer","reason":"通用定义无需检索","confidence":0.93,"arguments":{},"missing_slots":[]}')
    )
    hint = [{"source": "rag.md#definition", "matched_entities": ["rag"], "strong_match": True,
             "excerpt": "RAG combines retrieval and generation."}]
    decision = router.route("什么是 RAG", [], index_hint=hint)
    assert decision.mode == "direct_answer"
    assert "通用定义" in decision.reason


def test_repeated_domain_question_with_strong_entities_still_uses_rag() -> None:
    router = IntentRouter(FakeClient(
        '{"mode":"direct_answer","reason":"上轮已经回答，无需再次检索",'
        '"confidence":0.9,"arguments":{},"missing_slots":[]}'
    ))
    tools = [{
        "name": "knowledge.search",
        "description": "Search indexed knowledge.",
        "parameters": [{"name": "query", "type": "string", "required": True}],
    }]
    hint = [{
        "source": "falcon.md#fairness",
        "matched_entities": ["falcon", "scheduler"],
        "strong_match": True,
        "excerpt": "Falcon Scheduler fairness and overload recovery.",
    }]
    decision = router.route(
        "比较 Falcon Scheduler 的公平性与过载恢复策略",
        tools,
        memory_context="上一轮助手已经回答过相同问题。",
        index_hint=hint,
    )
    assert decision.mode == "tool_call"
    assert decision.tool_name == "knowledge.search"
    assert "历史回答不作为事实证据" in decision.reason


def test_router_rejects_knowledge_search_when_index_has_no_candidate() -> None:
    router = IntentRouter(FakeClient(
        '{"mode":"tool_call","reason":"技术问题应查知识库","tool_name":"knowledge.search",'
        '"arguments":{"query":"qwenVL 工作流程","top_k":5},"missing_slots":[],"confidence":0.9,'
        '"explicit_local_retrieval":false}'
    ))
    tools = [{
        "name": "knowledge.search",
        "description": "Search indexed knowledge.",
        "parameters": [{"name": "query", "type": "string", "required": True}],
    }]

    decision = router.route("qwenVL 的工作流程是什么样的？", tools, index_hint=[])

    assert decision.mode == "direct_answer"
    assert "没有与问题直接匹配的候选" in decision.reason


def test_explicit_local_retrieval_can_search_an_empty_candidate_set() -> None:
    router = IntentRouter(FakeClient(
        '{"mode":"tool_call","reason":"用户明确要求检查本地索引","tool_name":"knowledge.search",'
        '"arguments":{"query":"qwenVL 工作流程"},"missing_slots":[],"confidence":0.9,'
        '"explicit_local_retrieval":true}'
    ))
    tools = [{
        "name": "knowledge.search",
        "description": "Search indexed knowledge.",
        "parameters": [{"name": "query", "type": "string", "required": True}],
    }]

    decision = router.route("请检查当前知识库有没有 qwenVL 的工作流程", tools, index_hint=[])

    assert decision.mode == "tool_call"
    assert decision.tool_name == "knowledge.search"
    assert decision.explicit_local_retrieval is True


def test_router_rejects_unrequested_web_search_for_general_knowledge() -> None:
    router = IntentRouter(FakeClient(
        '{"mode":"tool_call","reason":"搜索可获得更多资料","tool_name":"web.search",'
        '"arguments":{"query":"qwenVL 工作流程"},"missing_slots":[],"confidence":0.8,'
        '"explicit_web_retrieval":false,"requires_fresh_information":false}'
    ))
    tools = [{
        "name": "web.search",
        "description": "Search the web.",
        "parameters": [{"name": "query", "type": "string", "required": True}],
    }]

    decision = router.route("qwenVL 的工作流程是什么样的？", tools, index_hint=[])

    assert decision.mode == "direct_answer"
    assert "不依赖实时信息" in decision.reason


def test_router_rechecks_false_explicit_web_claim() -> None:
    router = IntentRouter(FakeClient(
        '{"mode":"tool_call","reason":"联网补充模型架构","tool_name":"web.search",'
        '"arguments":{"query":"qwenVL workflow"},"missing_slots":[],"confidence":0.9,'
        '"explicit_web_retrieval":true,"requires_fresh_information":false}'
    ))
    tools = [{
        "name": "web.search",
        "description": "Search the web.",
        "parameters": [{"name": "query", "type": "string", "required": True}],
    }]
    with patch.object(router, "_verify_web_requirement", return_value=(False, False)):
        decision = router.route("qwenVL 的工作流程是什么样的？", tools, index_hint=[])
    assert decision.mode == "direct_answer"
    assert "不依赖实时信息" in decision.reason


def test_router_allows_web_search_for_fresh_information() -> None:
    router = IntentRouter(FakeClient(
        '{"mode":"tool_call","reason":"天气属于实时信息","tool_name":"web.search",'
        '"arguments":{"query":"明天上海天气"},"missing_slots":[],"confidence":0.9,'
        '"explicit_web_retrieval":false,"requires_fresh_information":true}'
    ))
    tools = [{
        "name": "web.search",
        "description": "Search the web.",
        "parameters": [{"name": "query", "type": "string", "required": True}],
    }]

    decision = router.route("明天上海天气怎么样？", tools, index_hint=[])

    assert decision.mode == "tool_call"
    assert decision.tool_name == "web.search"
    assert decision.requires_fresh_information is True


def test_router_recovers_empty_tool_call_to_direct_answer() -> None:
    router = IntentRouter(FakeClient(
        '{"mode":"tool_call","reason":"应直接回答但误写了模式","tool_name":"",'
        '"arguments":{},"missing_slots":[],"confidence":0.8,'
        '"explicit_local_retrieval":false,"explicit_web_retrieval":false,'
        '"requires_fresh_information":false}'
    ))

    decision = router.route("qwenVL 的工作流程是什么样的？", [], index_hint=[])

    assert decision.mode == "direct_answer"
    assert "结构一致性" in decision.reason


def test_offline_router_uses_strong_index_hint_for_domain_question() -> None:
    router = IntentRouter(FakeClient(""))
    tools = [{
        "name": "knowledge.search",
        "description": "Search indexed knowledge.",
        "parameters": [
            {"name": "query", "type": "string", "required": True},
            {"name": "top_k", "type": "integer", "required": False, "default": 5},
        ],
    }]
    hint = [{"source": "blip-2.pdf#page-3", "matched_entities": ["q-former"], "strong_match": True,
             "excerpt": "Q-Former bridges the frozen image encoder and language model."}]
    decision = router.route("Q-Former 如何连接冻结的视觉编码器和语言模型？", tools, index_hint=hint)
    assert decision.mode == "tool_call"
    assert decision.tool_name == "knowledge.search"
    assert decision.arguments["query"].startswith("Q-Former")


def test_intent_router_returns_clarify_when_required_slot_missing() -> None:
    router = IntentRouter(
        FakeClient(
            '{"mode":"tool_call","reason":"需要写文件","tool_name":"files.write_file","arguments":{"path":"D:\\\\tmp\\\\note.txt"},"missing_slots":[],"confidence":0.92}'
        )
    )
    tools = [
        {
            "name": "files.write_file",
            "parameters": [
                {"name": "path", "type": "string", "required": True},
                {"name": "content", "type": "string", "required": True},
            ],
        }
    ]

    decision = router.route("在 D:\\tmp\\note.txt 中写点内容", tools)

    assert decision.mode == "clarify"
    assert "content" in decision.missing_slots


def test_intent_router_does_not_ask_for_runtime_email_auth_status() -> None:
    router = IntentRouter(
        FakeClient(
            '{"mode":"clarify","reason":"需要确认邮箱认证","tool_name":"email.list_messages",'
            '"arguments":{"limit":5,"unread_only":true},"missing_slots":["email_auth_status"],"confidence":0.9}'
        )
    )
    tools = [
        {
            "name": "email.list_messages",
            "parameters": [
                {"name": "limit", "type": "integer", "required": False, "default": 20},
                {"name": "unread_only", "type": "boolean", "required": False, "default": False},
            ],
        }
    ]
    decision = router.route("帮我查看邮箱里最近的未读邮件", tools)
    assert decision.mode == "tool_call"
    assert decision.arguments["unread_only"] is True
    assert decision.missing_slots == []


def test_document_agent_uses_intent_router_for_generic_tool_call(base: Path) -> None:
    force_local_fallback()
    index = DocumentIndex(base / "index" / "index.json")
    agent = DocumentQAAgent(index)

    fake_client = FakeClient(
        '{"mode":"tool_call","reason":"执行演示工具","tool_name":"demo.echo","arguments":{"text":"hello"},"missing_slots":[],"confidence":0.95}'
    )
    agent.client = fake_client
    agent.intent_router.client = fake_client

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="demo.echo",
            category="demo",
            description="Echo the given text.",
            parameters=[ToolParameter("text", "string", True, "Text to echo")],
            handler=lambda text: {"echo": text},
        )
    )
    agent.tool_registry = registry

    result = agent.answer("请调用演示工具")

    assert result.answer
    assert "hello" in result.answer
    assert any(step.name == "route_intent" for step in result.steps)
    assert any(step.name == "execute_tool" for step in result.steps)
    assert any(step.name == "tool_result" for step in result.steps)


def test_agent_routes_q_former_to_existing_index_without_llm_router(base: Path) -> None:
    index = DocumentIndex(base / "qformer" / "index.json")
    document = Document(
        "blip2", str(base / "blip-2.pdf"), "blip-2.pdf", "pdf",
        "[Page 3]\nQ-Former connects the frozen visual encoder and frozen language model.",
        metadata={"sha256": "qformer-test"},
    )
    with patch.object(index.client, "embed", side_effect=lambda texts: [local_hash_embedding(x) for x in texts]):
        index.add_document(document)
    agent = DocumentQAAgent(index)
    with patch.object(agent.client, "chat", return_value=""), patch.object(
        index.client, "embed", side_effect=lambda texts: [local_hash_embedding(x) for x in texts]
    ), patch.object(agent.memory_store, "search", return_value=[]), patch.object(
        agent.memory_extractor, "extract", return_value=[]
    ):
        result = agent.answer("Q-Former 如何连接冻结的视觉编码器和语言模型？")
    assert result.evidences
    assert any(step.name == "index_route_hint" for step in result.steps)
    assert any(step.name == "route_intent" and "knowledge.search" in step.detail for step in result.steps)
    assert not any(step.name == "direct_answer" for step in result.steps)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="deskpilot_intent_test_") as tmp:
        base = Path(tmp)
        test_slot_validator_accepts_tool_dict_and_coerces_values()
        test_slot_validator_reports_missing_required_slots()
        test_intent_router_prefers_direct_answer()
        test_offline_fallback_routes_structural_file_write()
        test_offline_fallback_routes_structural_python_execution()
        test_index_hint_is_injected_but_does_not_force_generic_question()
        test_repeated_domain_question_with_strong_entities_still_uses_rag()
        test_router_rejects_knowledge_search_when_index_has_no_candidate()
        test_explicit_local_retrieval_can_search_an_empty_candidate_set()
        test_router_rejects_unrequested_web_search_for_general_knowledge()
        test_router_rechecks_false_explicit_web_claim()
        test_router_allows_web_search_for_fresh_information()
        test_router_recovers_empty_tool_call_to_direct_answer()
        test_offline_router_uses_strong_index_hint_for_domain_question()
        test_intent_router_returns_clarify_when_required_slot_missing()
        test_intent_router_does_not_ask_for_runtime_email_auth_status()
        test_document_agent_uses_intent_router_for_generic_tool_call(base)
        test_agent_routes_q_former_to_existing_index_without_llm_router(base)
    print("Intent router tests passed.")


if __name__ == "__main__":
    main()
