from __future__ import annotations

from pathlib import Path

from eval.release_validation import POOLS, case_mode, commands_for, load_cases, render_summary, select_cases


def test_release_pool_covers_every_module_and_never_selects_live_network() -> None:
    cases = load_cases()
    selected = select_cases(cases, "release-test-seed")
    assert len(selected) == 2 * len(POOLS)
    assert {row["subset"] for row in selected} == set(POOLS)
    assert all(row["mode"] in {"offline", "api"} for row in selected)
    assert all({row["situation"] for row in selected if row["subset"] == subset} == {"standard", "edge"} for subset in POOLS)


def test_release_pool_rotates_away_from_previous_where_alternatives_exist() -> None:
    cases = load_cases()
    first = select_cases(cases, "release-test-seed")
    second = select_cases(cases, "release-test-seed", {"selected": first})
    first_by_slot = {(row["subset"], row["situation"]): row["id"] for row in first}
    for row in second:
        pool = POOLS[row["subset"]][row["situation"]]
        available = [case_id for case_id in pool if case_mode(cases[case_id]) != "online"]
        if len(available) >= 2:
            assert row["id"] != first_by_slot[(row["subset"], row["situation"])]


def test_release_commands_split_offline_and_api_without_online(tmp_path: Path) -> None:
    selected = select_cases(load_cases(), "release-test-seed")
    commands = commands_for(selected, "0.9.1", tmp_path)
    assert "text_offline" in commands
    assert "text_api" in commands
    assert "multimodal_offline" in commands
    assert "multimodal_api" in commands
    assert all("online" not in command for command in commands.values())
    for label, command in commands.items():
        selected_ids = {row["id"] for row in selected if f"{row['family']}_{row['mode']}" == label}
        command_ids = {command[index + 1] for index, token in enumerate(command[:-1]) if token == "--case-id"}
        assert command_ids == selected_ids


def test_release_summary_never_counts_unrun_api_as_passed(tmp_path: Path) -> None:
    selected = select_cases(load_cases(), "release-test-seed")
    manifest = {
        "version": "0.9.1", "evaluation_target": "current_source_checkout",
        "artifact_sha256": None, "previous_manifest": None,
        "selected": selected, "commands": commands_for(selected, "0.9.1", tmp_path),
    }
    summary = render_summary(manifest, tmp_path)
    assert f"Selected: {len(selected)}" in summary
    assert f"not run: {len(selected)}" in summary
    assert "not_run" in summary
