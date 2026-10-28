from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import pytest

from deskpilot.core.agent import DocumentQAAgent
from deskpilot.core.models import Document
from deskpilot.intent.router import IntentRouter
from deskpilot.memory.policy import MemoryPolicy
from deskpilot.memory.memory_extractor import MemoryExtractor
from deskpilot.rag.chunker import ChunkingConfig, chunk_document_with_metadata
from deskpilot.rag.query_analyzer import QueryAnalyzer
from deskpilot.rag.vector_index import DocumentIndex
from deskpilot.tools.tool_registry import build_default_tool_registry


def test_release_current_directory_uses_authorized_workspace(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "authorized"
    workspace.mkdir()
    monkeypatch.setenv("DESKPILOT_RELEASE_MODE", "1")
    monkeypatch.setenv("DESKPILOT_HOME", str(tmp_path / "profile"))
    monkeypatch.setenv("TOOL_SAFE_ROOTS", str(workspace))
    index = DocumentIndex(tmp_path / "index" / "index.json")
    agent = DocumentQAAgent(index)
    assert agent.workspace_root == workspace
    request = agent._normalize_file_write_request(
        "帮我在当前目录下新建 hello.txt，并写入 hello",
        {"path": "hello.txt", "content": "hello"},
    )
    assert request["path"] == str(workspace / "hello.txt")
    assert agent.tool_registry.get_tool_spec("files.workspace_root").handler() == str(workspace)
    assert "shell.execute_command" not in {item["name"] for item in agent.tool_registry.list_tool_summaries()}
    created = agent.answer("帮我在当前目录下新建一个hello.txt，并写入一句hello")
    assert (workspace / "hello.txt").read_text(encoding="utf-8") == "hello"
    assert str(workspace / "hello.txt") in created.answer
    today = agent.answer("今天是几月几日")
    assert "当前本地日期和时间" in today.answer
    assert "shell.execute_command" not in today.answer
    deletion = agent.answer(f'删除这个文件"{workspace / "hello.txt"}"')
    assert deletion.pending_action is not None
    assert deletion.pending_action["tool_name"] == "files.delete_file"
    assert (workspace / "hello.txt").exists()
    approved = agent.approve_pending_action(deletion.pending_action, session_id=deletion.session_id)
    assert "已确认并删除文件" in approved.answer
    assert not (workspace / "hello.txt").exists()
    agent.task_service.close()


def test_release_time_and_file_delete_require_structured_approval(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DESKPILOT_RELEASE_MODE", "1")
    monkeypatch.setenv("DESKPILOT_HOME", str(tmp_path / "profile"))
    registry = build_default_tool_registry(None, workspace_root=tmp_path)
    now = registry.call("system.current_time")
    assert now.ok and "T" in now.output["local_time"]
    assert not registry.call_approved("shell.execute_command", command="date").ok

    target = tmp_path / "note.txt"
    target.write_text("original", encoding="utf-8")
    preview = registry.call("files.delete_file", path=str(target))
    assert preview.status == "waiting_human"
    assert target.exists()
    target.write_text("changed", encoding="utf-8")
    stale = registry.call_approved("files.delete_file", path=str(target),
                                   expected_sha256=preview.output["sha256"])
    assert not stale.ok and target.exists()
    fresh = registry.call("files.delete_file", path=str(target))
    approved = registry.call_approved("files.delete_file", path=str(target),
                                      expected_sha256=fresh.output["sha256"])
    assert approved.ok and approved.output["deleted"] and not target.exists()


def test_offline_router_uses_safe_time_and_delete_tools() -> None:
    router = IntentRouter.__new__(IntentRouter)
    tools = [{"name": "system.current_time"}, {"name": "files.delete_file"}]
    date_decision = router._fallback_route("今天是几月几日", tools, "")
    assert date_decision.tool_name == "system.current_time"
    delete_decision = router._fallback_route('删除这个文件"D:\\work\\hello.txt"', tools, "")
    assert delete_decision.tool_name == "files.delete_file"
    assert delete_decision.arguments["path"] == r"D:\work\hello.txt"


def test_semantic_chunker_batches_sentences_across_pages() -> None:
    document = Document(
        "two-pages", "paper.pdf", "paper.pdf", "pdf",
        "[Page 1]\n第一句。第二句。\n\n[Page 2]\n第三句。第四句。",
    )
    config = ChunkingConfig(policy="semantic", min_chunk_tokens=1, target_chunk_tokens=12,
                            max_chunk_tokens=24, parent_target_tokens=80, overlap_sentences=1)
    calls: list[list[str]] = []

    def embed(texts: list[str]) -> list[list[float]]:
        calls.append(list(texts))
        return [[1.0, 0.0] for _ in texts]

    result = chunk_document_with_metadata(document, config, embed)
    assert len(calls) == 1
    assert len(calls[0]) == 4
    assert len(result.parent_chunks) == 2


def test_folder_index_reports_per_file_progress_and_timing(tmp_path: Path) -> None:
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "a.md").write_text("# A\n这是第一篇文档。", encoding="utf-8")
    (folder / "b.txt").write_text("这是第二篇文档。", encoding="utf-8")
    index = DocumentIndex(tmp_path / "index" / "index.json")
    progress: list[tuple[int, int, str, dict]] = []
    result = index.add_folder(folder, progress=lambda done, total, path, trace:
                              progress.append((done, total, path.name, trace)))
    assert len(result) == 2
    assert [item[0] for item in progress] == [1, 2]
    assert all(item[1] == 2 and item[3]["duration_ms"] >= 0 for item in progress)
    assert {"chunking_ms", "embedding_ms", "catalog_ms", "save_ms"} <= progress[-1][3].keys()


def test_grounded_document_answer_is_not_duplicated_as_memory() -> None:
    policy = MemoryPolicy()
    step = type("Step", (), {"name": "retrieve_evidence"})()
    assert policy.should_extract("总结这篇论文", grounded=True, steps=[step]) == (
        False, "indexed_evidence_not_memory",
    )
    assert policy.should_extract("以后请先给结论", grounded=True, steps=[step]) == (
        True, "explicit_user_state",
    )


def test_completed_file_artifact_extracts_without_llm(monkeypatch) -> None:
    extractor = MemoryExtractor()
    monkeypatch.setattr(extractor.client, "chat", lambda *_args, **_kwargs: pytest.fail("Unexpected memory LLM call"))
    items = extractor.extract_artifacts(
        "帮我创建 hello.txt", "已写入文件：D:\\work\\hello.txt", "session", ["user", "assistant"],
    )
    assert len(items) == 1 and items[0]["memory_type"] == "artifact"
    assert "hello.txt" in items[0]["content"]


def test_weather_snippet_must_match_requested_forecast_date() -> None:
    assert DocumentQAAgent._snippet_mentions_date("10/27 明天 上海", date(2026, 10, 27))
    assert not DocumentQAAgent._snippet_mentions_date("10/14 旧新闻", date(2026, 10, 27))


def test_retrieval_query_drops_instruction_prefix() -> None:
    analysis = QueryAnalyzer().analyze("根据索引的知识库回答：多模态大模型有哪些常用方法")
    assert analysis.standalone_question == "多模态大模型有哪些常用方法"


def test_image_question_without_cloud_consent_returns_persisted_explanation(tmp_path: Path) -> None:
    from io import BytesIO
    from PIL import Image
    from deskpilot.multimodal.service import MultimodalService

    buffer = BytesIO()
    Image.new("RGB", (20, 20), (30, 40, 50)).save(buffer, format="PNG")
    index = DocumentIndex(tmp_path / "index" / "index.json")
    agent = DocumentQAAgent(index)
    agent.multimodal = MultimodalService(tmp_path / "multimodal")
    agent.multimodal.vision_llm.allow_cloud = False
    asset = agent.multimodal.add_chat_attachment_bytes(buffer.getvalue(), "sample.png")
    result = agent.answer_multimodal("描述一下这幅图中的内容", [asset.asset_id])
    assert "尚未允许" in result.answer
    assert "没有上传图片" in result.answer
    assert agent.session_store.read_messages(result.session_id)[-1].content == result.answer
    agent.task_service.close()


@pytest.mark.skipif(os.name != "nt", reason="Windows DPAPI only")
def test_release_email_settings_save_via_qt_bridge(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("DESKPILOT_RELEASE_MODE", "1")
    monkeypatch.setenv("DESKPILOT_HOME", str(tmp_path))
    from PySide6.QtCore import QObject
    from PySide6.QtWidgets import QApplication
    from deskpilot.qt_app import DeskPilotBridge
    from deskpilot.release.profile import read_settings, save_settings
    from deskpilot.release.secrets import SecretStore

    app = QApplication.instance() or QApplication([])
    save_settings({"QWEN_MODEL": "test-model"}, tmp_path)
    bridge = DeskPilotBridge.__new__(DeskPilotBridge)
    QObject.__init__(bridge)
    bridge._busy = False
    bridge._status = "就绪"
    bridge._release_info = {}
    assert bridge.saveEmailSettings("qq", "test@qq.com", "synthetic-authorization-code") is True
    assert read_settings(tmp_path)["EMAIL_PROVIDER"] == "qq"
    assert SecretStore(tmp_path / "secrets.dpapi").get("EMAIL_PASSWORD") == "synthetic-authorization-code"
    assert bridge.saveEmailSettings("qq", "test@qq.com", "") is True
    assert bridge.saveEmailSettings("163", "another@163.com", "") is False
    assert read_settings(tmp_path)["EMAIL_PROVIDER"] == "qq"
    app.processEvents()


@pytest.mark.skipif(os.name != "nt", reason="Windows DPAPI only")
def test_release_vision_settings_require_https_and_store_key_separately(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("DESKPILOT_RELEASE_MODE", "1")
    monkeypatch.setenv("DESKPILOT_HOME", str(tmp_path))
    from PySide6.QtCore import QObject
    from PySide6.QtWidgets import QApplication
    from deskpilot.qt_app import DeskPilotBridge
    from deskpilot.release.profile import read_settings
    from deskpilot.release.secrets import SecretStore

    app = QApplication.instance() or QApplication([])
    bridge = DeskPilotBridge.__new__(DeskPilotBridge)
    QObject.__init__(bridge)
    bridge._busy = False
    bridge._status = "ready"
    bridge._release_info = {}
    assert bridge.saveVisionSettings("vlm", "http://example.invalid/v1", "secret") is False
    assert read_settings(tmp_path) == {}
    assert bridge.saveVisionSettings("vlm", "https://example.invalid/v1", "synthetic-key") is True
    assert read_settings(tmp_path)["VISION_MODEL"] == "vlm"
    assert read_settings(tmp_path)["VISION_BASE_URL"] == "https://example.invalid/v1"
    assert SecretStore(tmp_path / "secrets.dpapi").get("VISION_API_KEY") == "synthetic-key"
    app.processEvents()


@pytest.mark.skipif(os.name != "nt", reason="Windows DPAPI only")
def test_unreadable_vault_requires_explicit_reset_and_preserves_old_bytes(tmp_path: Path) -> None:
    from deskpilot.release.secrets import SecretStore

    path = tmp_path / "secrets.dpapi"
    path.write_bytes(b"synthetic-invalid-dpapi")
    vault = SecretStore(path)
    with pytest.raises(RuntimeError, match="无法解密"):
        vault.set("EMAIL_PASSWORD", "replacement")
    assert path.read_bytes() == b"synthetic-invalid-dpapi"
    previous = vault.reset()
    assert previous is not None and previous.read_bytes() == b"synthetic-invalid-dpapi"
    vault.set("EMAIL_PASSWORD", "replacement")
    assert vault.get("EMAIL_PASSWORD") == "replacement"


@pytest.mark.skipif(os.name != "nt", reason="Windows DPAPI only")
def test_email_settings_reports_unreadable_vault_without_changing_settings(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DESKPILOT_RELEASE_MODE", "1")
    monkeypatch.setenv("DESKPILOT_HOME", str(tmp_path))
    from PySide6.QtCore import QObject
    from PySide6.QtWidgets import QApplication
    from deskpilot.qt_app import DeskPilotBridge
    from deskpilot.release.profile import read_settings, save_settings

    app = QApplication.instance() or QApplication([])
    save_settings({"EMAIL_PROVIDER": "mock"}, tmp_path)
    vault = tmp_path / "secrets.dpapi"
    vault.write_bytes(b"synthetic-invalid-dpapi")
    bridge = DeskPilotBridge.__new__(DeskPilotBridge)
    QObject.__init__(bridge)
    bridge._busy = False
    bridge._status = "ready"
    bridge._release_info = {}
    errors: list[str] = []
    bridge.errorRaised.connect(errors.append)
    assert bridge.saveEmailSettings("qq", "test@qq.com", "synthetic-code") is False
    assert "邮箱设置保存失败" in errors[-1]
    assert "重置凭据" in errors[-1]
    assert bridge._status == "邮箱设置保存失败"
    assert read_settings(tmp_path)["EMAIL_PROVIDER"] == "mock"
    assert vault.read_bytes() == b"synthetic-invalid-dpapi"
    bridge._busy = True
    assert bridge.saveEmailSettings("qq", "test@qq.com", "synthetic-code") is False
    assert "当前任务仍在运行" in errors[-1]
    app.processEvents()
