"""Deterministic fault injection. Never connects to real mail or model services."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from deskpilot.core.approval import PendingActionStore
from deskpilot.multi_agent.plan_executor import PlanExecutor
from deskpilot.multi_agent.schemas import TaskPlan, TaskPlanStep
from deskpilot.multi_agent.supervisor import SupervisorAgent
from deskpilot.multi_agent.task_ledger import TaskLedger
from deskpilot.tasks.control import ExecutionControl, TaskCancelled, TaskPaused, execution_scope
from deskpilot.tasks.retry import RetryPolicy, UnknownSideEffect
from deskpilot.tasks.service import TaskService
from deskpilot.tasks.store import TaskStore
from deskpilot.tools.tool_registry import ToolRegistry, ToolSpec


SCENARIOS = (
    "nonzero_exit", "false_result", "transient_retry", "auth_no_retry", "bounded_retry",
    "deadline", "cancel_process", "approval_restart", "attachment_changed",
    "checkpoint_process_restart", "unknown_no_replay", "pdf_pause_resume",
    "file_compensation", "file_compensation_conflict", "mail_dispatch_reconcile", "attachment_renewal_guard",
)


def _executor(ledger: TaskLedger) -> PlanExecutor:
    return PlanExecutor(tool_call=lambda *a, **k: None, research=lambda q: None,
                        llm_call=lambda q: "fixture", task_ledger=ledger,
                        supervisor=SupervisorAgent(retry_policy=RetryPolicy(base_delay=0, jitter=0)))


def run_scenario(scenario: str, root: Path, seed: int = 0) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    result: dict = {"scenario": scenario, "seed": seed, "real_api": False}
    if scenario in {"nonzero_exit", "false_result"}:
        registry = ToolRegistry()
        payload = {"executed": True, "returncode": 3, "stderr": "fixture failure"} if scenario == "nonzero_exit" else {"ok": False}
        registry.register(ToolSpec("fixture", "test", "test", [], lambda: payload))
        output = registry.call("fixture")
        assert not output.ok and output.status == "failed"
        result.update(state=output.status, attempts=1)
    elif scenario in {"transient_retry", "auth_no_retry", "bounded_retry", "deadline"}:
        count = 0
        target = 2 + seed % 2

        def handler(values):
            nonlocal count
            count += 1
            if scenario == "auth_no_retry":
                raise PermissionError("fixture 401")
            if scenario == "bounded_retry" or count < target:
                raise ConnectionError("fixture 503")
            return {"ok": True}

        control = ExecutionControl(retry_policy=RetryPolicy(max_attempts=3, base_delay=0, jitter=0))
        if scenario == "deadline":
            control.deadline = time.monotonic() - 1
        plan = TaskPlan("fixture", "single", [TaskPlanStep("n", "knowledge", "fixture", handler=handler, max_retries=2)])
        output = SupervisorAgent().execute(plan, control=control)
        expected = {"auth_no_retry": 1, "bounded_retry": 3, "deadline": 0}.get(scenario, target)
        assert count == expected
        assert output.status == ("success" if scenario == "transient_retry" else "failed")
        result.update(state=output.status, attempts=count, retries=control.usage["retries"])
    elif scenario == "cancel_process":
        from deskpilot.tools.execution_tools import _run_process
        marker = root / "started"
        code = f"from pathlib import Path\nimport time\nPath({str(marker)!r}).write_text('started')\ntime.sleep(20)"
        control = ExecutionControl()
        cancellation_times = []

        def cancel():
            end = time.monotonic() + 5
            while not marker.exists() and time.monotonic() < end:
                time.sleep(0.01)
            cancellation_times.append(time.monotonic())
            control.cancel()

        thread = threading.Thread(target=cancel)
        thread.start()
        try:
            with execution_scope(control):
                _run_process([sys.executable, "-c", code], root, 10, 2048)
            raise AssertionError("cancel signal was ignored")
        except TaskCancelled:
            pass
        finally:
            thread.join(timeout=6)
        assert marker.exists()
        result.update(state="cancelled", attempts=1, cancel_latency_seconds=time.monotonic() - cancellation_times[0])
    elif scenario in {"approval_restart", "attachment_changed"}:
        path = root / "attachment.txt"
        path.write_text("approved content", encoding="utf-8")
        db = root / "approvals.sqlite3"
        first = PendingActionStore(db_path=db)
        public = first.create("session", {"tool_name": "email.send", "kwargs": {
            "to": ["recipient@example.com"], "subject": "fixture", "body": "fixture", "attachment_paths": [str(path)]}})
        restored = PendingActionStore(db_path=db)
        if scenario == "attachment_changed":
            path.write_text("changed after approval", encoding="utf-8")
            try:
                restored.consume(public["action_id"], "session")
                raise AssertionError("changed attachment accepted")
            except ValueError:
                pass
            assert restored.inspect(public["action_id"], "session") is not None
            result.update(state="waiting_human", submissions=0)
        else:
            try:
                restored.consume(public["action_id"], "foreign")
            except PermissionError:
                pass
            else:
                raise AssertionError("cross-session approval accepted")
            item = restored.consume(public["action_id"], "session")
            assert item.kwargs["body"] == "fixture"
            try:
                restored.consume(public["action_id"], "session")
                raise AssertionError("approval replay accepted")
            except ValueError:
                pass
            result.update(state="consumed", duplicate_submissions=0)
    elif scenario == "checkpoint_process_restart":
        db = root / "ledger.sqlite3"
        marker = root / "source_calls.txt"
        script = '''import os, sys
from pathlib import Path
from deskpilot.multi_agent.task_ledger import TaskLedger
from deskpilot.multi_agent.plan_executor import PlanExecutor
root=Path(sys.argv[1])
ledger=TaskLedger(root/'ledger.sqlite3')
def source(values):
    marker=root/'source_calls.txt'
    marker.write_text(marker.read_text()+'1' if marker.exists() else '1')
    return {'evidence':'saved evidence'}
def crash(values):
    os._exit(23)
plan={'goal':'fixture','steps':[{'id':'source','agent':'knowledge','max_retries':1},
{'id':'report','agent':'knowledge','depends_on':['source'],'max_retries':1}]}
PlanExecutor(tool_call=lambda *a,**k:None,research=lambda q:None,llm_call=lambda q:'fixture',task_ledger=ledger).execute_bound_plan(plan,{'source':source,'report':crash})
'''
        env = dict(os.environ)
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
        process = subprocess.run([sys.executable, "-c", script, str(root)], env=env,
                                 capture_output=True, timeout=15)
        assert process.returncode == 23, process.stderr.decode(errors="replace")
        ledger = TaskLedger(db)
        task = ledger.list_tasks()[0]
        calls = 0

        def source(values):
            nonlocal calls
            calls += 1
            return {"evidence": "incorrect replay"}

        plan = {"goal": "fixture", "steps": [{"id": "source", "agent": "knowledge", "max_retries": 1},
                {"id": "report", "agent": "knowledge", "depends_on": ["source"], "max_retries": 1}]}
        output = _executor(ledger).execute_bound_plan(plan, {"source": source, "report": lambda values: {"report": values["evidence"]}},
                                                    resume_task_id=task["task_id"])
        assert output.status == "success", output.error
        assert calls == 0 and marker.read_text() == "1" and output.values["report"] == "saved evidence"
        result.update(state="success", checkpoint_reused=1, checkpoint_eligible=1, duplicate_submissions=0)
    elif scenario == "unknown_no_replay":
        store = TaskStore(root / "tasks.sqlite3")
        service = TaskService(store)
        service.register("fixture", lambda p: None)
        task_id = service.submit("fixture", {})
        store.save_plan(task_id, {"goal": "fixture", "steps": [{"id": "send", "agent": "commit", "side_effect": True}]})
        store.update_node(task_id, "send", "unknown", record_attempt=False)
        store.finish_task(task_id, "recovery_required")
        try:
            service.resume(task_id)
            raise AssertionError("unknown side effect was replayed")
        except ValueError:
            pass
        result.update(state="recovery_required", automatic_replays=0)
    elif scenario == "pdf_pause_resume":
        import fitz
        from deskpilot.multimodal.asset_store import AssetStore
        from deskpilot.multimodal.ingestion import MultimodalIngestionService
        from deskpilot.multimodal.providers.ocr import OCRRegion
        from deskpilot.multimodal.vector_store import MultimodalVectorStore
        control = ExecutionControl()

        class OCR:
            provider = "fixture"
            calls = 0

            def recognize_many(self, paths):
                self.calls += len(paths)
                if self.calls == 1:
                    control.pause()
                return [[OCRRegion("page evidence", 1.0)] for p in paths]

        class Vision:
            embedding_space = "fixture:vision:2"
            model_id = "fixture"
            model_revision = "1"

            def embed_images(self, paths):
                return [[1.0, 0.0] for p in paths]

        class Text:
            embedding_space = "fixture:text:2"
            model_id = "fixture"
            model_revision = "1"

            def embed(self, texts):
                return [[1.0, 0.0] for t in texts]

        pdf = root / "fixture.pdf"
        with fitz.open() as document:
            for n in range(3):
                page = document.new_page()
                page.insert_text((30, 30), f"Fixture page {n}")
            document.save(pdf)
        ocr = OCR()
        ingestion = MultimodalIngestionService(AssetStore(root / "assets"), MultimodalVectorStore(root / "vectors.sqlite3"),
                                               ocr, Vision(), Text())
        previous = os.environ.get("MULTIMODAL_PDF_BATCH_SIZE")
        os.environ["MULTIMODAL_PDF_BATCH_SIZE"] = "1"
        try:
            with execution_scope(control):
                try:
                    ingestion.ingest_pdf(pdf)
                    raise AssertionError("pause was ignored")
                except TaskPaused:
                    pass
            control.resume()
            with execution_scope(control):
                output = ingestion.ingest_pdf(pdf)
            assert len(output) == 3 and ocr.calls == 3
            assert output[0]["checkpoint_reused"]
            result.update(state="success", checkpoint_reused=1, checkpoint_eligible=1)
        finally:
            if previous is None:
                os.environ.pop("MULTIMODAL_PDF_BATCH_SIZE", None)
            else:
                os.environ["MULTIMODAL_PDF_BATCH_SIZE"] = previous
    elif scenario in {"file_compensation", "file_compensation_conflict"}:
        from deskpilot.tasks.compensation import compensate_file_write
        from deskpilot.tasks.effects import begin_effect, effects_for_task, finish_effect
        from deskpilot.tools.tool_registry import ToolResult
        from deskpilot.tools.write_tools import write_file
        target = root / "report.txt"
        target.write_text("before", encoding="utf-8")
        store = TaskStore(root / "tasks.sqlite")
        control = ExecutionControl(task_id="fixture-file", node_id="write")
        control.task_store = store
        with execution_scope(control):
            op = begin_effect(control, "files.write_file", {"path": str(target), "content": "after", "overwrite": True})
            output = write_file(target, "after", overwrite=True, confirm=True, safe_roots=[root])
            finish_effect(control, op, ToolResult(True, "files.write_file", output))
        assert compensate_file_write(store, op, safe_roots=[root])["permission"]["requires_confirmation"]
        if scenario == "file_compensation_conflict":
            target.write_text("external edit", encoding="utf-8")
            try:
                compensate_file_write(store, op, confirm=True, safe_roots=[root])
                raise AssertionError("external edit was overwritten")
            except RuntimeError:
                pass
            assert target.read_text(encoding="utf-8") == "external edit"
            assert effects_for_task(store, "fixture-file")[0]["state"] == "committed"
            result.update(state="conflict_protected")
        else:
            compensate_file_write(store, op, confirm=True, safe_roots=[root])
            assert target.read_text(encoding="utf-8") == "before"
            assert effects_for_task(store, "fixture-file")[0]["state"] == "compensated"
            result.update(state="compensated")
    elif scenario == "mail_dispatch_reconcile":
        from deskpilot.mcp.email_mcp import ImapSmtpEmailProvider
        from deskpilot.tasks.effects import begin_effect, effects_for_task
        from deskpilot.tasks.mail_reconcile import reconcile_email_effect
        store = TaskStore(root / "tasks.sqlite")
        control = ExecutionControl(task_id="fixture-mail", node_id="send")
        control.task_store = store
        provider = ImapSmtpEmailProvider.__new__(ImapSmtpEmailProvider)
        provider.username = "fixture@example.com"
        with execution_scope(control):
            op = begin_effect(control, "email.send", {"to": ["fixture@example.com"]})
            provider._persist_mime(provider._build_message(["fixture@example.com"], "subject", "body"))
        with execution_scope(control):
            try:
                begin_effect(control, "email.send", {"to": ["fixture@example.com"]})
                raise AssertionError("SMTP dispatch replayed")
            except RuntimeError as exc:
                assert "interrupted after dispatch" in str(exc)
        class ReadOnlyProvider:
            def find_sent_copy(self, message_id):
                assert message_id.startswith("<deskpilot-")
                return False
            def send(self, *args, **kwargs):
                raise AssertionError("SMTP must not be retried")
        assert reconcile_email_effect(store, ReadOnlyProvider(), op)["status"] == "unknown"
        assert effects_for_task(store, "fixture-mail")[0]["state"] == "dispatched"
        result.update(state="manual_review", automatic_replays=0)
    elif scenario == "attachment_renewal_guard":
        attachment = root / "attachment.txt"
        attachment.write_text("v1", encoding="utf-8")
        approvals = PendingActionStore(db_path=root / "tasks.sqlite")
        item = approvals.create("fixture", {"tool_name": "email.save_draft", "description": "fixture draft",
                                             "kwargs": {"to": ["fixture@example.com"], "subject": "fixture",
                                                        "body": "body", "attachment_paths": [str(attachment)]}})
        assert "attachment.txt" in item["preview"]
        attachment.write_text("v2", encoding="utf-8")
        try:
            approvals.renew(item["action_id"], "fixture")
            raise AssertionError("changed attachment renewed")
        except ValueError:
            pass
        assert approvals.inspect(item["action_id"], "fixture") is None
        result.update(state="approval_invalidated")
    else:
        raise ValueError(f"Unknown reliability scenario: {scenario}")
    result.update(passed=True, response_seconds=time.monotonic() - started)
    return result
