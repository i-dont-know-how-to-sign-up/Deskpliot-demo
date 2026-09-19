from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from deskpilot.mcp.email_mcp import EmailMCPService, MockEmailProvider
from deskpilot.tools.tool_registry import build_default_tool_registry
from deskpilot.core.agent import DocumentQAAgent
from deskpilot.core.runtime import PlanAndExecuteRuntime
from deskpilot.core.models import AgentStep
from deskpilot.rag.vector_index import DocumentIndex
from deskpilot.intent.schemas import IntentDecision
import tempfile


def test_mock_email_read_classify_search_and_draft() -> None:
    service = EmailMCPService(MockEmailProvider())
    assert service.list_messages(10)["count"] == 2
    assert service.search("评审")["count"] == 1
    assert service.classify(10)["待回复"]
    draft = service.draft(["alice@example.com"], "Re: 评审", "收到")
    assert draft["requires_confirmation"] is True


def test_email_write_operations_require_confirmation() -> None:
    provider = MockEmailProvider()
    service = EmailMCPService(provider)
    pending = service.send(["a@example.com"], "test", "body")
    assert pending["permission"]["requires_confirmation"] is True
    assert provider.sent == []
    sent = service.send(["a@example.com"], "test", "body", confirm=True)
    assert sent["status"] == "sent"
    assert len(provider.sent) == 1


def test_email_tools_are_registered_and_protected() -> None:
    service = EmailMCPService(MockEmailProvider())
    registry = build_default_tool_registry(None, email_service=service)
    names = {item["name"] for item in registry.list_tools(category="email")}
    assert {"email.list_messages", "email.search", "email.send"}.issubset(names)
    pending = registry.call("email.send", to=["a@example.com"], subject="test", body="body")
    assert pending.ok is True
    assert pending.output["permission"]["requires_confirmation"] is True


def test_unread_email_request_forces_unread_only() -> None:
    """即使路由器漏填参数，明确的“未读”语义也必须传给邮箱工具。"""
    with tempfile.TemporaryDirectory(prefix="deskpilot_email_test_") as tmp:
        agent = DocumentQAAgent(DocumentIndex(Path(tmp) / "index.json"))
        agent.tool_registry = build_default_tool_registry(
            None, email_service=EmailMCPService(MockEmailProvider())
        )
        normalized = agent._normalize_email_arguments(
            "帮我查看邮箱里的未读邮件",
            "email.list_messages",
            {"limit": 20, "unread_only": False},
        )
        result = agent.tool_registry.call("email.list_messages", **normalized)
        assert result.ok is True
        assert result.output["count"] == 1
        assert result.output["messages"][0]["unread"] is True


def test_failed_tool_result_is_not_rendered_as_empty_success() -> None:
    """工具异常必须在最终回答中保留原因。"""
    with tempfile.TemporaryDirectory(prefix="deskpilot_email_test_") as tmp:
        agent = DocumentQAAgent(DocumentIndex(Path(tmp) / "index.json"))
        result = agent._finalize_generic_tool_result(
            question="查看未读邮件",
            tool_name="email.list_messages",
            tool_result=type("FailedResult", (), {"ok": False, "output": None, "error": "邮箱认证失败"})(),
            call_arguments={"unread_only": True},
            steps=[],
            session_id="test",
            user_message_id="user",
            memory_context=type("Context", (), {"text": ""})(),
        )
        assert "邮箱认证失败" in result.answer
        assert any(step.status == "failed" for step in result.steps)


def test_email_provider_label_is_not_hardcoded_to_163() -> None:
    service = EmailMCPService(MockEmailProvider())
    from deskpilot.mcp.email_mcp import ImapSmtpEmailProvider

    provider = ImapSmtpEmailProvider(
        {
            "imap_host": "imap.qq.com",
            "smtp_host": "smtp.qq.com",
            "username": "test@qq.com",
            "password": "test-code",
        }
    )
    assert provider._provider_label() == "QQ"


def test_html_email_body_is_converted_to_plain_text() -> None:
    from email.message import EmailMessage as MIMEMessage
    from deskpilot.mcp.email_mcp import _body

    message = MIMEMessage()
    message.set_content("<html><style>.x{color:red}</style><div>标题</div><p>正文 &amp; 详情</p></html>", subtype="html")
    body = _body(message)
    assert "标题" in body
    assert "正文 & 详情" in body
    assert "<div>" not in body
    assert "color:red" not in body


def test_mock_recent_messages_are_sorted_by_date() -> None:
    from deskpilot.mcp.email_mcp import EmailMessage

    provider = MockEmailProvider(
        [
            EmailMessage("old", "t1", "a@example.com", [], "old", "old", "2026-01-01", True),
            EmailMessage("new", "t2", "b@example.com", [], "new", "new", "2026-09-10", True),
        ]
    )
    messages = provider.list_messages(2, unread_only=True)
    assert [item.message_id for item in messages] == ["new", "old"]


def test_approved_email_draft_is_reported_as_success() -> None:
    with tempfile.TemporaryDirectory(prefix="deskpilot_email_test_") as tmp:
        provider = MockEmailProvider()
        agent = DocumentQAAgent(DocumentIndex(Path(tmp) / "index.json"))
        agent.tool_registry = build_default_tool_registry(
            None, email_service=EmailMCPService(provider)
        )
        result = agent.approve_pending_action(
            {
                "tool_name": "email.save_draft",
                "kwargs": {"to": ["a@example.com"], "subject": "test", "body": "body"},
                "description": "保存邮件草稿",
            }
        )
        assert "保存邮件草稿" in result.answer
        assert any(step.name == "execute_approved_action" and step.status == "success" for step in result.steps)


def test_email_recipients_are_normalized_and_empty_addresses_rejected() -> None:
    service = EmailMCPService(MockEmailProvider())
    preview = service.send("Alice <alice@example.com>", "test", "body", confirm=False)
    assert preview["permission"]["requires_confirmation"] is True
    try:
        service.send(["", "bad-address"], "test", "body", confirm=True)
    except ValueError as exc:
        assert "收件人地址格式错误" in str(exc)
    else:
        raise AssertionError("invalid recipients must be rejected before provider execution")


def test_email_attachment_is_validated_and_replayed_after_confirmation() -> None:
    with tempfile.TemporaryDirectory(prefix="deskpilot_attachment_") as tmp:
        attachment = Path(tmp) / "agentic_rl.md"
        attachment.write_text("# Agentic RL\n\nPaper summary", encoding="utf-8")
        provider = MockEmailProvider()
        service = EmailMCPService(provider)

        pending = service.send(
            ["a@example.com"], "agentic RL", "请查收附件", [str(attachment)]
        )
        assert pending["permission"]["requires_confirmation"] is True
        assert pending["attachments"][0]["name"] == "agentic_rl.md"
        assert provider.sent == []

        sent = service.send(
            ["a@example.com"], "agentic RL", "请查收附件", [str(attachment)], confirm=True
        )
        assert sent["status"] == "sent"
        assert sent["attachment_paths"] == [str(attachment.resolve())]


def test_smtp_message_contains_real_mime_attachment() -> None:
    from deskpilot.mcp.email_mcp import ImapSmtpEmailProvider

    with tempfile.TemporaryDirectory(prefix="deskpilot_mime_") as tmp:
        attachment = Path(tmp) / "report.md"
        attachment.write_text("报告正文", encoding="utf-8")
        provider = ImapSmtpEmailProvider(
            {
                "imap_host": "imap.example.com",
                "smtp_host": "smtp.example.com",
                "username": "me@example.com",
                "password": "secret",
            }
        )
        message = provider._build_message(
            ["a@example.com"], "report", "body", [str(attachment)]
        )
        attachments = list(message.iter_attachments())
        assert len(attachments) == 1
        assert attachments[0].get_filename() == "report.md"
        assert attachments[0].get_payload(decode=True) == attachment.read_bytes()


def test_missing_email_attachment_is_rejected_before_approval() -> None:
    service = EmailMCPService(MockEmailProvider())
    try:
        service.send(["a@example.com"], "test", "body", ["missing-report.md"])
    except FileNotFoundError as exc:
        assert "邮件附件不存在" in str(exc)
    else:
        raise AssertionError("missing attachment must be rejected before approval")


def test_research_email_passes_generated_report_to_send_approval() -> None:
    """验证调研产物不会在 Planner 到邮件工具之间丢失。"""
    with tempfile.TemporaryDirectory(prefix="deskpilot_research_mail_") as tmp:
        report_path = Path(tmp) / "agentic_rl.md"
        report_path.write_text("# Agentic RL\n\n- Paper A", encoding="utf-8")

        class FakeResearchAgent:
            def research(self, topic: str):
                return type(
                    "ResearchResult",
                    (),
                    {
                        "report": report_path.read_text(encoding="utf-8"),
                        "artifact_path": str(report_path),
                        "used_llm": False,
                    },
                )()

        class FakeClient:
            def chat(self, messages, temperature=0.2):
                return "邮件简介：调研包含 Paper A，详细内容见附件。"

        agent = DocumentQAAgent.__new__(DocumentQAAgent)
        agent.web_research_agent = FakeResearchAgent()
        agent.client = FakeClient()
        agent.runtime = PlanAndExecuteRuntime()
        agent.tool_registry = build_default_tool_registry(
            None, email_service=EmailMCPService(MockEmailProvider())
        )
        agent._finalize_generic_tool_result = lambda **kwargs: kwargs

        result = agent._answer_research_email_request(
            question="调研 Agentic RL 并作为附件发送",
            request={
                "to": "a@example.com",
                "subject": "agentic RL",
                "request": "Agentic RL 最新论文",
                "attach_report": True,
            },
            steps=[],
            session_id="s",
            user_message_id="m",
            memory_context=object(),
        )
        assert result["call_arguments"]["attachment_paths"] == [str(report_path)]
        assert result["tool_result"].output["attachments"][0]["name"] == "agentic_rl.md"
