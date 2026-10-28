from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="DeskPilot private single-tenant API")
    parser.add_argument("--index", type=Path, required=True, help="Index shared with the dedicated worker")
    parser.add_argument("--tokens", type=Path, required=True, help="JSON file with SHA-256 bearer token digests")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "::1", "localhost"}:
        parser.error("API binds only to loopback; use an authenticated TLS gateway for private-network access")
    if not 1 <= args.port <= 65535:
        parser.error("Invalid port")
    os.environ["DESKPILOT_RELEASE_MODE"] = "1"
    from ..release.profile import configure_runtime
    configure_runtime()
    from .app import create_app
    from .security import TokenCatalog
    from .service import PrivateTaskService
    import uvicorn

    service = PrivateTaskService(args.index,
                                 sandbox_enabled=os.getenv("DESKPILOT_ENABLE_DOCKER_SANDBOX") == "1")
    app = create_app(service, TokenCatalog(args.tokens))
    uvicorn.run(app, host=args.host, port=args.port, access_log=False, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
