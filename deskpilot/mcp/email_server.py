from __future__ import annotations

from .email_mcp import create_mcp_server


def main() -> None:
    # MCP 客户端通过 stdio 启动本文件，不在 stdout 输出凭证或调试日志。
    create_mcp_server().run(transport="stdio")


if __name__ == "__main__":
    main()
