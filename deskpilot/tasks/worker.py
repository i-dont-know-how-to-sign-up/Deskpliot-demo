"""Optional headless worker; the same SQLite lease prevents a second UI worker."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="DeskPilot durable task worker")
    parser.add_argument("--index", type=Path, help="Use an isolated document index")
    parser.add_argument("--once", action="store_true", help="Execute at most one queued task")
    parser.add_argument("--list", action="store_true", help="List persisted task summaries only")
    args = parser.parse_args()
    from ..rag.vector_index import DocumentIndex
    from ..core.agent import DocumentQAAgent
    agent = DocumentQAAgent(DocumentIndex(args.index) if args.index else DocumentIndex())
    service = agent.task_service
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
