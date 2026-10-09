from __future__ import annotations

from enum import StrEnum


class StepStatus(StrEnum):
    """Agent、Supervisor 和 UI 共享的步骤状态契约。"""

    SUCCESS = "success"
    FAILED = "failed"
    WAITING_HUMAN = "waiting_human"
    PENDING = "pending"
    RETRY = "retry"
    SKIPPED = "skipped"


class ToolNames:
    """跨 Router、Planner 与执行器使用的稳定工具标识。"""

    KNOWLEDGE_SEARCH = "knowledge.search"
    KNOWLEDGE_ANSWER_MULTIMODAL = "knowledge.answer_multimodal"
    WEB_SEARCH = "web.search"
    WEB_RESEARCH = "web.research"
    FILES_READ_DOCUMENT = "files.read_document"
    FILES_WRITE_FILE = "files.write_file"
    EMAIL_SEND = "email.send"
    EMAIL_SAVE_DRAFT = "email.save_draft"
    SHELL_EXECUTE_COMMAND = "shell.execute_command"
    CODE_EXECUTE_PYTHON = "code.execute_python"
    VISION_ANSWER_ATTACHMENTS = "vision.answer_attachments"


SOURCE_TOOLS = frozenset({
    ToolNames.WEB_SEARCH,
    ToolNames.WEB_RESEARCH,
    ToolNames.KNOWLEDGE_SEARCH,
    ToolNames.FILES_READ_DOCUMENT,
})

COMMIT_TOOLS = frozenset({
    ToolNames.FILES_WRITE_FILE,
    ToolNames.EMAIL_SEND,
    ToolNames.EMAIL_SAVE_DRAFT,
})
