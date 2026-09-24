from __future__ import annotations

import email
import html
import imaplib
import json
import mimetypes
import os
import re
import smtplib
import ssl
import re
from email.utils import getaddresses
import time
from dataclasses import asdict, dataclass, field
from email.header import decode_header, make_header
from email.message import EmailMessage as MIMEEmailMessage
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Protocol


@dataclass
class EmailMessage:
    """统一不同邮箱服务商返回的邮件结构。"""

    message_id: str
    thread_id: str
    sender: str
    recipients: list[str]
    subject: str
    body: str
    date: str = ""
    unread: bool = False
    labels: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        data = asdict(self)
        data["body"] = self.body[:2000]
        return data


class EmailProvider(Protocol):
    def list_messages(self, limit: int = 20, unread_only: bool = False) -> list[EmailMessage]: ...
    def read_thread(self, thread_id: str) -> list[EmailMessage]: ...
    def search(self, query: str, limit: int = 20) -> list[EmailMessage]: ...
    def create_draft(
        self, to: list[str], subject: str, body: str, attachment_paths: list[str] | None = None
    ) -> dict[str, Any]: ...
    def send(
        self, to: list[str], subject: str, body: str, attachment_paths: list[str] | None = None
    ) -> dict[str, Any]: ...


def _decode(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    try:
        return str(make_header(decode_header(value)))
    except Exception:
        return str(value)


def _body(message: email.message.Message) -> str:
    if message.is_multipart():
        parts = [part for part in message.walk() if part.get_content_type() == "text/plain"]
        for part in parts:
            payload = part.get_payload(decode=True)
            if payload:
                charset = part.get_content_charset() or "utf-8"
                return payload.decode(charset, errors="replace")
        # 许多营销邮件只有 HTML 正文；优先转换成纯文本，不能返回原始标签。
        for part in message.walk():
            if part.get_content_type() != "text/html":
                continue
            payload = part.get_payload(decode=True)
            if payload:
                charset = part.get_content_charset() or "utf-8"
                return _html_to_text(payload.decode(charset, errors="replace"))
        return ""
    payload = message.get_payload(decode=True)
    if isinstance(payload, bytes):
        content = payload.decode(message.get_content_charset() or "utf-8", errors="replace")
        if message.get_content_type() == "text/html":
            return _html_to_text(content)
        return content
    return str(payload or "")


def _html_to_text(value: str) -> str:
    """将 HTML 邮件转换成适合展示和交给 LLM 的纯文本。"""
    value = re.sub(r"(?s)<!--.*?-->", " ", value)
    value = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", value)
    value = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</tr>|</li>|</h[1-6]>", "\n", value)
    value = re.sub(r"(?s)<[^>]+>", " ", value)
    value = html.unescape(value)
    lines = [re.sub(r"[ \t\r\f\v]+", " ", line).strip() for line in value.splitlines()]
    # 超长带查询参数 URL 多为跟踪链接；保留短链接和正文中的可读地址。
    lines = [line for line in lines if not (line.startswith(("http://", "https://")) and len(line) > 300)]
    return "\n".join(line for line in lines if line)


class MockEmailProvider:
    """无网络邮箱后端，供开发和评测使用。"""

    def __init__(self, messages: list[EmailMessage] | None = None):
        self.messages = messages or [
            EmailMessage("mock-1", "thread-1", "alice@example.com", ["me@example.com"], "项目评审时间确认", "请确认明天 10:00 是否可以参加评审。", "2026-09-10", True, ["inbox"]),
            EmailMessage("mock-2", "thread-2", "bob@example.com", ["me@example.com"], "周报提交提醒", "请在周五前提交本周周报。", "2026-09-09", False, ["inbox"]),
        ]
        self.drafts: list[dict[str, Any]] = []
        self.sent: list[dict[str, Any]] = []

    def list_messages(self, limit: int = 20, unread_only: bool = False) -> list[EmailMessage]:
        items = [item for item in self.messages if not unread_only or item.unread]
        return sorted(items, key=lambda item: item.date or "", reverse=True)[: max(1, min(limit, 100))]

    def read_thread(self, thread_id: str) -> list[EmailMessage]:
        return [item for item in self.messages if item.thread_id == thread_id]

    def search(self, query: str, limit: int = 20) -> list[EmailMessage]:
        needle = query.casefold()
        return [item for item in self.messages if needle in f"{item.subject} {item.body} {item.sender}".casefold()][:limit]

    def create_draft(
        self, to: list[str], subject: str, body: str, attachment_paths: list[str] | None = None
    ) -> dict[str, Any]:
        draft = {
            "to": to,
            "subject": subject,
            "body": body,
            "attachment_paths": list(attachment_paths or []),
            "status": "draft",
        }
        self.drafts.append(draft)
        return draft

    def send(
        self, to: list[str], subject: str, body: str, attachment_paths: list[str] | None = None
    ) -> dict[str, Any]:
        sent = {
            "to": to,
            "subject": subject,
            "body": body,
            "attachment_paths": list(attachment_paths or []),
            "status": "sent",
        }
        self.sent.append(sent)
        return sent


class ImapSmtpEmailProvider:
    """163、QQ 和支持 IMAP/SMTP 的 Outlook 账户适配器。"""

    def __init__(self, config: dict[str, Any]):
        self.host = str(config["imap_host"])
        self.imap_port = int(config.get("imap_port", 993))
        self.smtp_host = str(config["smtp_host"])
        self.smtp_port = int(config.get("smtp_port", 465))
        self.username = str(config["username"])
        self.password = str(config["password"])
        self.mailbox = str(config.get("mailbox", "INBOX"))
        self.drafts_mailbox = str(config.get("drafts_mailbox", "Drafts"))
        self.sent_mailbox = str(config.get("sent_mailbox", "Sent"))
        self.timeout = max(5, min(int(config.get("timeout_seconds", 20)), 120))

    def _connect(self, mailbox: str | None = None, readonly: bool = True) -> imaplib.IMAP4_SSL:
        target_mailbox = mailbox or self.mailbox
        try:
            client = imaplib.IMAP4_SSL(
                self.host,
                self.imap_port,
                ssl_context=ssl.create_default_context(),
                timeout=self.timeout,
            )
            status, detail = client.login(self.username, self.password)
            if status != "OK":
                raise RuntimeError(f"IMAP 登录失败：{detail!r}")
            status, detail = client.select(target_mailbox, readonly=readonly)
            if status != "OK" and readonly and "Unsafe Login" in str(detail):
                # 部分 163 节点会拒绝只读 EXAMINE，但允许 SELECT。
                # SELECT 本身不会把邮件标记为已读，读取正文时再使用 BODY.PEEK 保持只读语义。
                status, detail = client.select(target_mailbox, readonly=False)
            if status != "OK":
                raise RuntimeError(f"无法打开邮箱文件夹 {target_mailbox}：{detail!r}")
        except Exception as exc:
            detail = str(exc)
            if "Unsafe Login" in detail:
                provider = self._provider_label()
                detail = (
                    f"{provider} 邮箱拒绝了当前 IMAP 登录（Unsafe Login）。请先在邮箱网页端完成安全验证，"
                    "确认已开启 IMAP/SMTP，并重新生成客户端授权码；EMAIL_PASSWORD 必须填写授权码而不是网页登录密码。"
                    f"如果仍被拦截，请联系 {provider} 官方客服解除客户端登录限制。"
                )
            raise RuntimeError(
                f"IMAP 连接邮箱失败（服务器={self.host}:{self.imap_port}，文件夹={target_mailbox}）：{detail}"
            ) from exc
        return client

    def _provider_label(self) -> str:
        """根据 IMAP 主机生成用户可读的服务商名称，避免错误提示写死为 163。"""
        labels = {
            "imap.qq.com": "QQ",
            "imap.163.com": "163",
            "outlook.office365.com": "Outlook",
        }
        return labels.get(self.host.lower(), self.host)

    def _append_to_mailbox(self, message: MIMEEmailMessage, mailbox: str, flags: str) -> bool:
        """将邮件副本写入 IMAP 文件夹，供网页版邮箱显示。"""
        client = self._connect(mailbox, readonly=False)
        try:
            status, detail = client.append(
                mailbox,
                flags,
                imaplib.Time2Internaldate(time.time()),
                message.as_bytes(),
            )
            if status != "OK":
                raise RuntimeError(f"写入 IMAP 文件夹 {mailbox} 失败：{detail!r}")
            return True
        finally:
            try:
                client.logout()
            except Exception:
                pass

    def _parse(self, raw: bytes, uid: str, unread: bool) -> EmailMessage:
        message = email.message_from_bytes(raw)
        sender = _decode(message.get("From"))
        recipients = [_decode(value.strip()) for value in message.get("To", "").split(",") if value.strip()]
        date = message.get("Date", "")
        try:
            date = parsedate_to_datetime(date).isoformat()
        except Exception:
            pass
        thread_id = _decode(message.get("Thread-Index") or message.get("References") or uid)
        return EmailMessage(uid, thread_id, sender, recipients, _decode(message.get("Subject")), _body(message), date, unread, [self.mailbox.lower()])

    def list_messages(self, limit: int = 20, unread_only: bool = False) -> list[EmailMessage]:
        client = self._connect()
        try:
            criteria = "UNSEEN" if unread_only else "ALL"
            status, data = client.uid("search", None, criteria)
            if status != "OK":
                raise RuntimeError(f"IMAP 搜索邮件失败：{data!r}")
            uids = data[0].split() if data and data[0] else []
            result = []
            for uid in uids:
                status, fetched = client.uid("fetch", uid, "(BODY.PEEK[])")
                if status != "OK":
                    raise RuntimeError(f"读取邮件 UID {uid!r} 失败：{fetched!r}")
                raw = next((item[1] for item in fetched if isinstance(item, tuple)), b"")
                if raw:
                    result.append(self._parse(raw, uid.decode(), unread_only))
            # UID 通常递增，但不能把 UID 当成时间；按邮件 Date 排序后再截取最近邮件。
            result.sort(key=lambda item: item.date or "", reverse=True)
            return result[: max(1, min(limit, 100))]
        finally:
            try:
                client.logout()
            except Exception:
                pass

    def read_thread(self, thread_id: str) -> list[EmailMessage]:
        return [item for item in self.search(thread_id, 100) if item.thread_id == thread_id or thread_id in item.thread_id]

    def search(self, query: str, limit: int = 20) -> list[EmailMessage]:
        client = self._connect()
        try:
            _status, data = client.uid("search", None, f'(OR SUBJECT "{query}" BODY "{query}")')
            result = []
            for uid in (data[0].split() if data and data[0] else [])[-limit:]:
                _status, fetched = client.uid("fetch", uid, "(RFC822)")
                raw = next((item[1] for item in fetched if isinstance(item, tuple)), b"")
                if raw:
                    result.append(self._parse(raw, uid.decode(), False))
            return result
        finally:
            client.logout()

    def _build_message(
        self, to: list[str], subject: str, body: str, attachment_paths: list[str] | None = None
    ) -> MIMEEmailMessage:
        """构造正文和附件共用的 MIME 邮件，保证草稿与实际发送内容一致。"""
        message = MIMEEmailMessage()
        message["From"] = self.username
        message["To"] = ", ".join(to)
        message["Subject"] = subject
        message.set_content(body)
        for raw_path in attachment_paths or []:
            path = Path(raw_path)
            mime_type, _encoding = mimetypes.guess_type(path.name)
            maintype, subtype = (mime_type or "application/octet-stream").split("/", 1)
            message.add_attachment(
                path.read_bytes(),
                maintype=maintype,
                subtype=subtype,
                filename=path.name,
            )
        return message

    def _send(
        self,
        to: list[str],
        subject: str,
        body: str,
        send: bool,
        attachment_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        message = self._build_message(to, subject, body, attachment_paths)
        if self.smtp_port == 465:
            client = smtplib.SMTP_SSL(self.smtp_host, self.smtp_port, context=ssl.create_default_context())
        else:
            client = smtplib.SMTP(self.smtp_host, self.smtp_port)
            client.starttls(context=ssl.create_default_context())
        with client:
            client.login(self.username, self.password)
            if send:
                client.send_message(message)
        if send:
            # SMTP 投递不会自动保证 IMAP 已发送箱出现副本，因此单独 APPEND。
            try:
                sent_copy_saved = self._append_to_mailbox(message, self.sent_mailbox, "(\\Seen)")
            except Exception as exc:
                sent_copy_saved = False
                sent_copy_error = str(exc)
            else:
                sent_copy_error = ""
            return {
                "to": to,
                "subject": subject,
                "body": body,
                "attachment_paths": list(attachment_paths or []),
                "status": "sent",
                "sent_copy_saved": sent_copy_saved,
                "sent_copy_error": sent_copy_error,
            }
        return {
            "to": to,
            "subject": subject,
            "body": body,
            "attachment_paths": list(attachment_paths or []),
            "status": "draft_preview",
        }

    def create_draft(
        self, to: list[str], subject: str, body: str, attachment_paths: list[str] | None = None
    ) -> dict[str, Any]:
        message = self._build_message(to, subject, body, attachment_paths)
        self._append_to_mailbox(message, self.drafts_mailbox, "(\\Draft)")
        return {
            "to": to,
            "subject": subject,
            "body": body,
            "attachment_paths": list(attachment_paths or []),
            "status": "draft_saved",
            "mailbox": self.drafts_mailbox,
        }

    def send(
        self, to: list[str], subject: str, body: str, attachment_paths: list[str] | None = None
    ) -> dict[str, Any]:
        return self._send(to, subject, body, send=True, attachment_paths=attachment_paths)


class EmailMCPService:
    """MCP 工具的业务层；工具注册器和可选 MCP Server 共用此层。"""

    def __init__(self, provider: EmailProvider | None = None):
        self.provider = provider or self._from_env()

    def _from_env(self) -> EmailProvider:
        provider = os.getenv("EMAIL_PROVIDER", "mock").lower()
        if provider == "mock":
            path = os.getenv("EMAIL_MOCK_FILE", "")
            if path and Path(path).exists():
                data = json.loads(Path(path).read_text(encoding="utf-8"))
                return MockEmailProvider([EmailMessage(**item) for item in data])
            return MockEmailProvider()
        preset = {
            "163": ("imap.163.com", "smtp.163.com"),
            "qq": ("imap.qq.com", "smtp.qq.com"),
            "outlook": ("outlook.office365.com", "smtp.office365.com"),
        }
        imap_host, smtp_host = preset.get(provider, (os.getenv("EMAIL_IMAP_HOST", ""), os.getenv("EMAIL_SMTP_HOST", "")))
        required = {"EMAIL_USERNAME": os.getenv("EMAIL_USERNAME"), "EMAIL_PASSWORD": os.getenv("EMAIL_PASSWORD"), "EMAIL_IMAP_HOST": imap_host, "EMAIL_SMTP_HOST": smtp_host}
        missing = [key for key, value in required.items() if not value]
        if missing:
            raise RuntimeError(f"Missing email configuration: {', '.join(missing)}")
        return ImapSmtpEmailProvider({"imap_host": imap_host, "imap_port": os.getenv("EMAIL_IMAP_PORT", "993"), "smtp_host": smtp_host, "smtp_port": os.getenv("EMAIL_SMTP_PORT", "465"), "username": required["EMAIL_USERNAME"], "password": required["EMAIL_PASSWORD"], "mailbox": os.getenv("EMAIL_MAILBOX", "INBOX"), "drafts_mailbox": os.getenv("EMAIL_DRAFTS_MAILBOX", "Drafts"), "sent_mailbox": os.getenv("EMAIL_SENT_MAILBOX", "Sent")})

    def list_messages(self, limit: int = 20, unread_only: bool = False) -> dict[str, Any]:
        messages = self.provider.list_messages(limit, unread_only)
        return {"messages": [item.summary() for item in messages], "count": len(messages)}

    def read_thread(self, thread_id: str) -> dict[str, Any]:
        messages = self.provider.read_thread(thread_id)
        return {"thread_id": thread_id, "messages": [item.summary() for item in messages], "count": len(messages)}

    def search(self, query: str, limit: int = 20) -> dict[str, Any]:
        messages = self.provider.search(query, limit)
        return {"query": query, "messages": [item.summary() for item in messages], "count": len(messages)}

    def classify(self, limit: int = 20) -> dict[str, Any]:
        items = self.provider.list_messages(limit)
        groups: dict[str, list[dict[str, Any]]] = {"待回复": [], "待办": [], "通知": [], "其他": []}
        for item in items:
            text = f"{item.subject} {item.body}"
            category = "待回复" if any(word in text for word in ("回复", "确认", "请问", "是否")) else "待办" if any(word in text for word in ("提交", "截止", "完成", "提醒")) else "通知" if any(word in text for word in ("通知", "公告")) else "其他"
            groups[category].append(item.summary())
        return groups

    def summarize(self, thread_id: str) -> dict[str, Any]:
        messages = self.provider.read_thread(thread_id)
        return {"thread_id": thread_id, "summary": "\n".join(f"- {item.sender}: {item.subject}；{item.body[:300]}" for item in messages), "message_count": len(messages)}

    def draft(self, to: list[str], subject: str, body: str, template: str = "") -> dict[str, Any]:
        to = self._normalize_recipients(to)
        if template:
            body = template.replace("{body}", body).replace("{subject}", subject)
        return {"to": to, "subject": subject, "body": body, "status": "draft", "requires_confirmation": True}

    def save_draft(
        self,
        to: list[str],
        subject: str,
        body: str,
        attachment_paths: list[str] | None = None,
        confirm: bool = False,
    ) -> dict[str, Any]:
        to = self._normalize_recipients(to)
        paths, attachments = self._validate_attachments(attachment_paths)
        if not confirm:
            return {
                "permission": {
                    "requires_confirmation": True,
                    "risk_level": "medium",
                    "reasons": ["Saving a draft changes mailbox state."],
                },
                "attachments": attachments,
                "action": "save_draft",
            }
        result = dict(self.provider.create_draft(to, subject, body, paths))
        # 统一 Mock 和 IMAP 后端的成功状态，便于审批回放和 UI 使用同一契约。
        result["status"] = "draft_saved"
        return result

    def send(
        self,
        to: list[str],
        subject: str,
        body: str,
        attachment_paths: list[str] | None = None,
        confirm: bool = False,
    ) -> dict[str, Any]:
        to = self._normalize_recipients(to)
        paths, attachments = self._validate_attachments(attachment_paths)
        if not confirm:
            reasons = ["Sending an email has external side effects."]
            if attachments:
                reasons.append(f"The email contains {len(attachments)} local attachment(s).")
            return {
                "permission": {
                    "requires_confirmation": True,
                    "risk_level": "high",
                    "reasons": reasons,
                },
                "attachments": attachments,
                "action": "send",
            }
        return self.provider.send(to, subject, body, paths)

    def _validate_attachments(
        self, raw_paths: list[str] | str | None
    ) -> tuple[list[str], list[dict[str, Any]]]:
        """在审批前校验附件，避免确认后才发现路径或大小非法。"""
        if not raw_paths:
            return [], []
        values = [raw_paths] if isinstance(raw_paths, str) else list(raw_paths)
        max_file_bytes = max(1, int(os.getenv("EMAIL_ATTACHMENT_MAX_MB", "20"))) * 1024 * 1024
        max_total_bytes = max(1, int(os.getenv("EMAIL_ATTACHMENTS_TOTAL_MAX_MB", "25"))) * 1024 * 1024
        normalized: list[str] = []
        metadata: list[dict[str, Any]] = []
        total_size = 0
        for value in values:
            path = Path(str(value)).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"邮件附件不存在：{path}")
            if not path.is_file():
                raise ValueError(f"邮件附件必须是文件：{path}")
            size = path.stat().st_size
            if size > max_file_bytes:
                raise ValueError(f"邮件附件超过单文件大小限制：{path.name} ({size} bytes)")
            total_size += size
            if total_size > max_total_bytes:
                raise ValueError(f"邮件附件总大小超过限制：{total_size} bytes")
            path_text = str(path)
            if path_text in normalized:
                continue
            normalized.append(path_text)
            metadata.append({"name": path.name, "path": path_text, "size": size})
        return normalized, metadata

    def _normalize_recipients(self, raw: list[str] | str) -> list[str]:
        """统一处理字符串/列表/显示名地址，并在 SMTP 调用前拒绝空地址。"""
        values = [raw] if isinstance(raw, str) else list(raw or [])
        candidates: list[str] = []
        for value in values:
            text = str(value or "").strip()
            if not text:
                continue
            parsed = getaddresses([text])
            candidates.extend(address.strip() for _, address in parsed if address.strip())
        unique = list(dict.fromkeys(candidates))
        invalid = [address for address in unique if not re.fullmatch(r"[^@\s<>]+@[^@\s<>]+\.[^@\s<>]+", address)]
        if invalid or not unique:
            detail = ", ".join(invalid) if invalid else "空收件人"
            raise ValueError(f"收件人地址格式错误：{detail}")
        return unique


def create_mcp_server(service: EmailMCPService | None = None):
    """创建 MCP stdio Server；mcp 包按需安装，核心 Demo 不强制依赖它。"""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise RuntimeError("MCP server requires: python -m pip install mcp") from exc
    email = service or EmailMCPService()
    server = FastMCP("deskpilot-email")
    server.tool()(email.list_messages)
    server.tool()(email.read_thread)
    server.tool()(email.search)
    server.tool()(email.classify)
    server.tool()(email.summarize)
    server.tool()(email.draft)
    server.tool()(email.save_draft)
    server.tool()(email.send)
    return server
