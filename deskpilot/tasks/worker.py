"""Optional headless worker; the same SQLite lease prevents a second UI worker."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="DeskPilot durable task worker")
    parser.add_argument("--index", type=Path, help="Use an isolated document index")
    parser.add_argument("--once", action="store_true", help="Execute at most one queued task")
    parser.add_argument("--list", action="store_true", help="List persisted task summaries only")
    parser.add_argument("--api", action="store_true", help="Dedicated private-API worker; code and shell stay disabled")
    args = parser.parse_args()
    if args.api:
        if args.index is None:
            parser.error("--api requires an explicit --index")
        os.environ["DESKPILOT_RELEASE_MODE"] = "1"
        os.environ["DESKPILOT_API_WORKER"] = "1"
        from ..release.profile import configure_runtime
        configure_runtime()
    from ..rag.vector_index import DocumentIndex
    from ..core.agent import DocumentQAAgent
    agent = DocumentQAAgent(DocumentIndex(args.index) if args.index else DocumentIndex())
    if args.api:
        from ..api.worker import register_api_handlers
        register_api_handlers(agent)
    service = agent.task_service
    if args.api:
        service.owner = "api_" + service.owner
    try:
        if args.list:
            print(json.dumps(agent.list_tasks(), ensure_ascii=False, indent=2))
        elif args.once:
            service.run_once()
        else:
            service.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        service.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
