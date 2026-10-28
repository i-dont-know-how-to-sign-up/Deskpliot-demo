"""Select rotating release cases without touching live services by default."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from deskpilot import __version__

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "text": ROOT / "eval/dataset/deskpilot_bench.jsonl",
    "multimodal": ROOT / "eval/dataset/multimodal_cases.jsonl",
    "api_http": ROOT / "eval/dataset/p2_api_cases.jsonl",
}
DEFAULT_OUTPUT = ROOT / "eval/release_runs"

# Two independent situations per module. "edge" includes denied, missing, and
# constrained-input cases; it is not an expected test failure.
POOLS: dict[str, dict[str, tuple[str, ...]]] = {
    "DirectQA": {"standard": ("direct_001", "direct_002"), "edge": ("direct_005", "direct_006")},
    "RAG-DocQA": {"standard": ("rag_001", "rag_003"), "edge": ("ragas_003", "reg_p1_negative_001")},
    "Intent-Routing": {"standard": ("route_002", "route_003"), "edge": ("route_005", "route_012")},
    "Tool-Calling": {"standard": ("tool_007", "tool_009"), "edge": ("tool_005", "tool_008")},
    "Web-Research": {"standard": ("p2_web_fallback_001", "api_web_medium_001"), "edge": ("api_web_complex_001", "web_008")},
    "Memory": {"standard": ("memory_001", "memory_002"), "edge": ("memory_005", "memory_008")},
    "Context-Engineering": {"standard": ("context_p3_002", "context_002"), "edge": ("context_001", "context_004")},
    "Multi-Agent": {"standard": ("multi_001", "multi_002"), "edge": ("multi_003", "multi_005")},
    "Multi-Agent-P1": {"standard": ("multi_p1_001", "multi_exec_001"), "edge": ("multi_p1_002", "multi_p1_003")},
    "Composite-Workflow": {"standard": ("composite_001", "composite_002"), "edge": ("composite_005", "composite_006")},
    "Safety-Permission": {"standard": ("safe_004", "safe_007"), "edge": ("safe_002", "safe_005")},
    "Recovery": {"standard": ("recovery_001", "recovery_004"), "edge": ("recovery_002", "recovery_003")},
    "Multimodal-Safety": {"standard": ("mm_safe_001", "mm_safe_007"), "edge": ("mm_safe_002", "mm_safe_003")},
    "Multimodal-RAG": {"standard": ("mm_rag_001", "mm_rag_002"), "edge": ("mm_rag_003", "mm_rag_004")},
    "Multimodal-QA": {"standard": ("mm_qa_001", "mm_qa_002"), "edge": ("mm_qa_003", "mm_qa_004")},
    "Multimodal-Memory": {"standard": ("mm_memory_002", "mm_memory_003"), "edge": ("mm_memory_004", "mm_memory_006")},
    "P2-API": {"standard": ("p2_api_001", "p2_api_003", "p2_api_005", "p2_api_007"),
               "edge": ("p2_api_002", "p2_api_004", "p2_api_006", "p2_api_008")},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_cases() -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    for family, path in DATASETS.items():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            case = json.loads(line)
            case_id = str(case["id"])
            if case_id in cases:
                raise ValueError(f"Duplicate evaluation case ID: {case_id}")
            cases[case_id] = {**case, "family": family}
    return cases


def case_mode(case: dict[str, Any]) -> str:
    runtime = case.get("runtime") or {}
    if runtime.get("requires_network") or runtime.get("requires_online") or runtime.get("requires_email"):
        return "online"
    if runtime.get("requires_llm") or runtime.get("requires_vlm"):
        return "api"
    return "offline"


def select_cases(cases: dict[str, dict[str, Any]], seed: str, previous: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    previous_ids = {str(item["id"]) for item in (previous or {}).get("selected", [])}
    selected: list[dict[str, Any]] = []
    for subset, situations in POOLS.items():
        for situation, candidates in situations.items():
            eligible = []
            for case_id in candidates:
                case = cases.get(case_id)
                if case is None or case.get("subset") != subset:
                    raise ValueError(f"Invalid release pool case: {subset}/{situation}/{case_id}")
                if case_mode(case) != "online":
                    eligible.append(case_id)
            if not eligible:
                raise ValueError(f"No safe release cases: {subset}/{situation}")
            fresh = [case_id for case_id in eligible if case_id not in previous_ids]
            choices = fresh or eligible
            rng = random.Random(hashlib.sha256(f"{seed}:{subset}:{situation}".encode()).digest())
            case_id = rng.choice(choices)
            case = cases[case_id]
            selected.append({
                "id": case_id, "subset": subset, "situation": situation,
                "family": case["family"], "mode": case_mode(case),
                "repeated_from_previous": case_id in previous_ids,
            })
    return selected


def latest_manifest(output_root: Path) -> Path | None:
    manifests = sorted(output_root.glob("*/manifest.json"), key=lambda path: path.stat().st_mtime)
    return manifests[-1] if manifests else None


def commands_for(selected: list[dict[str, Any]], version: str, run_dir: Path) -> dict[str, list[str]]:
    commands: dict[str, list[str]] = {}
    for family, module in (("text", "eval.run_eval"), ("multimodal", "eval.run_multimodal_eval"),
                           ("api_http", "eval.run_p2_api_eval")):
        for mode in ("offline", "api"):
            ids = [row["id"] for row in selected if row["family"] == family and row["mode"] == mode]
            if not ids:
                continue
            label = f"{family}_{mode}"
            command = [sys.executable, "-m", module, "--mode", mode, "--project-version", version]
            for case_id in ids:
                command.extend(("--case-id", case_id))
            command.extend(("--output", str(run_dir / f"{label}.jsonl"), "--report", str(run_dir / f"{label}.md")))
            commands[label] = command
    return commands


def render_summary(manifest: dict[str, Any], run_dir: Path) -> str:
    outcomes: dict[str, str] = {}
    for label in manifest["commands"]:
        path = run_dir / f"{label}.jsonl"
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    result = json.loads(line)
                    outcomes[str(result.get("case_id"))] = str(result.get("status", "error"))
    lines = [
        f"# DeskPilot {manifest['version']} release validation", "",
        f"- Target: {manifest['evaluation_target']}",
        f"- Build SHA-256: {manifest['artifact_sha256'] or 'not supplied'}",
        f"- Previous manifest: {manifest['previous_manifest'] or 'none'}",
        f"- Selected: {len(manifest['selected'])}",
        f"- Executed: {len(outcomes)}; passed: {sum(value == 'passed' for value in outcomes.values())}; "
        f"failed/error/skipped: {sum(value not in {'passed'} for value in outcomes.values())}; "
        f"not run: {len(manifest['selected']) - len(outcomes)}", "",
        "| Module | Situation | Case ID | Mode | Result | Repeated |",
        "|---|---|---|---|---|---|",
    ]
    for row in manifest["selected"]:
        lines.append(
            f"| {row['subset']} | {row['situation']} | {row['id']} | {row['mode']} | "
            f"{outcomes.get(row['id'], 'not_run')} | {'yes' if row['repeated_from_previous'] else 'no'} |"
        )
    lines.extend(["", "`not_run` is not a pass. API/VLM checks require explicit execution and a separate result review."])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare rotating release checks; real API checks are never run automatically.")
    parser.add_argument("--version", default=__version__)
    parser.add_argument("--artifact", type=Path, help="Packaged executable whose SHA-256 identifies this build")
    parser.add_argument("--previous-manifest", type=Path, help="Override automatic previous-run lookup")
    parser.add_argument("--refresh-manifest", type=Path, help="Refresh the summary after manually running saved API commands")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-offline", action="store_true", help="Run selected offline cases only")
    args = parser.parse_args()

    if args.refresh_manifest:
        if not args.refresh_manifest.is_file():
            parser.error(f"Manifest is not a file: {args.refresh_manifest}")
        manifest = json.loads(args.refresh_manifest.read_text(encoding="utf-8"))
        summary_path = args.refresh_manifest.parent / "summary.md"
        summary_path.write_text(render_summary(manifest, args.refresh_manifest.parent), encoding="utf-8")
        print(f"Release summary: {summary_path}")
        return 0

    if args.artifact and not args.artifact.is_file():
        parser.error(f"Artifact is not a file: {args.artifact}")
    if args.version != __version__:
        print(f"Warning: report version {args.version} differs from source version {__version__}; this does not change the built package.")
    previous_path = args.previous_manifest or latest_manifest(args.output_root)
    previous = json.loads(previous_path.read_text(encoding="utf-8")) if previous_path else None
    cases = load_cases()
    dataset_hashes = {family: sha256_file(path) for family, path in DATASETS.items()}
    artifact_hash = sha256_file(args.artifact) if args.artifact else None
    seed = hashlib.sha256(json.dumps({
        "version": args.version, "artifact_sha256": artifact_hash,
        "datasets": dataset_hashes,
    }, sort_keys=True).encode()).hexdigest()
    selected = select_cases(cases, seed, previous)
    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%f%z")
    run_dir = args.output_root / f"deskpilot_release_v{args.version}_{stamp}_{seed[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    commands = commands_for(selected, args.version, run_dir)
    manifest = {
        "version": args.version, "created_at": datetime.now().astimezone().isoformat(),
        "evaluation_target": "current_source_checkout",
        "artifact": str(args.artifact.resolve()) if args.artifact else None,
        "artifact_sha256": artifact_hash, "dataset_sha256": dataset_hashes,
        "seed": seed, "previous_manifest": str(previous_path) if previous_path else None,
        "selected": selected, "commands": commands, "runs": {},
    }
    if args.run_offline:
        temp_dir = run_dir / "temp"
        temp_dir.mkdir()
        env = {**os.environ, "TMP": str(temp_dir), "TEMP": str(temp_dir)}
        for label, command in commands.items():
            if not label.endswith("_offline"):
                continue
            completed = subprocess.run(command, cwd=ROOT, env=env, check=False)
            rows = [json.loads(line) for line in (run_dir / f"{label}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()] if (run_dir / f"{label}.jsonl").exists() else []
            expected_ids = {row["id"] for row in selected if f"{row['family']}_{row['mode']}" == label}
            actual_ids = {str(row.get("case_id")) for row in rows}
            manifest["runs"][label] = {
                "exit_code": completed.returncode, "expected": len(expected_ids),
                "executed": len(rows), "passed": sum(row.get("status") == "passed" for row in rows),
                "skipped": sum(row.get("status") == "skipped" for row in rows),
                "gate_passed": completed.returncode == 0 and actual_ids == expected_ids and all(row.get("status") == "passed" for row in rows),
            }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run_dir / "summary.md").write_text(render_summary(manifest, run_dir), encoding="utf-8")
    print(f"Release manifest: {run_dir / 'manifest.json'}")
    for label, command in commands.items():
        print(f"{label}: {subprocess.list2cmdline(command)}")
    if args.run_offline:
        for label, result in manifest["runs"].items():
            print(f"{label}: {result['passed']}/{result['expected']} passed, gate={result['gate_passed']}")
        return 0 if all(result["gate_passed"] for result in manifest["runs"].values()) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
