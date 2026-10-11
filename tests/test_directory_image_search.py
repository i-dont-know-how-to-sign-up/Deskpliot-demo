from __future__ import annotations

from pathlib import Path
import json

import pytest
from PIL import Image

from deskpilot.core.agent import DocumentQAAgent
from deskpilot.core.plan_repair import PlanRepairResult
from deskpilot.intent.schemas import IntentDecision
from deskpilot.intent.router import IntentRouter
from deskpilot.multi_agent.planner import PlannerAgent
from deskpilot.multimodal.service import MultimodalService
from deskpilot.multimodal.providers.vision_embedding import BasicVisualEmbeddingProvider
from deskpilot.rag.web_research import WebResearchAgent
from deskpilot.rag.vector_index import DocumentIndex
from deskpilot.tools.tool_registry import ToolParameter, ToolRegistry, ToolSpec, build_default_tool_registry


class ColorVision:
    provider = "siglip2"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def embed_images(self, paths: list[Path]) -> list[list[float]]:
        vectors = []
        for path in paths:
            with Image.open(path) as image:
                red, blue = image.convert("RGB").getpixel((0, 0))[::2]
            vectors.append([1.0, 0.0] if red > blue else [0.0, 1.0])
        return vectors


def test_directory_image_search_is_scoped_and_does_not_index(tmp_path: Path) -> None:
    directory = tmp_path / "dogs"
    directory.mkdir()
    Image.new("RGB", (8, 8), "blue").save(directory / "blue.jpg")
    Image.new("RGB", (8, 8), "red").save(directory / "red.jpg")
    Image.new("RGB", (8, 8), "red").save(tmp_path / "outside.jpg")
    (directory / "broken.jpg").write_bytes(b"not an image")
    service = MultimodalService(tmp_path / "data")
    service.vision_embedding = ColorVision()
    before = service.vector_store.stats()

    result = service.search_directory_images(directory, "red dog", top_k=2)

    assert result["scanned"] == 2
    assert result["skipped"] == 1
    assert result["matches"][0]["path"] == str((directory / "red.jpg").resolve())
    assert all(str(directory.resolve()) in item["path"] for item in result["matches"])
    assert service.vector_store.stats() == before


def test_directory_image_search_reports_missing_model_and_bounded_directory(
    tmp_path: Path, monkeypatch,
) -> None:
    directory = tmp_path / "dogs"
    directory.mkdir()
    for index in range(2):
        Image.new("RGB", (8, 8), "red").save(directory / f"dog_{index}.jpg")
    service = MultimodalService(tmp_path / "data")
    service.vision_embedding = BasicVisualEmbeddingProvider()
    with pytest.raises(RuntimeError, match="siglip2"):
        service.search_directory_images(directory, "red dog")
    service.vision_embedding = ColorVision()
    monkeypatch.setenv("IMAGE_DIRECTORY_SEARCH_MAX_FILES", "1")
    with pytest.raises(ValueError, match="上限"):
        service.search_directory_images(directory, "red dog")


def test_planner_accepts_directory_image_search_node() -> None:
    planner = PlannerAgent()
    from deskpilot.multi_agent.schemas import TaskPlan, TaskPlanStep
    plan = TaskPlan(
        goal="find a red dog", route="single_agent", complexity_score=3,
        steps=[TaskPlanStep(
            "knowledge", "knowledge", "Find the image",
            allowed_tools=["knowledge.search_directory_images"],
            arguments={"directory": "C:\\photos", "query": "red dog"},
        )],
    )
    assert planner.validate(plan) == (True, "")


def test_registered_directory_image_tool_returns_scoped_matches(tmp_path: Path) -> None:
    directory = tmp_path / "dogs"
    directory.mkdir()
    Image.new("RGB", (8, 8), "red").save(directory / "red.jpg")
    index = DocumentIndex(tmp_path / "index" / "index.json")
    service = MultimodalService(tmp_path / "data")
    service.vision_embedding = ColorVision()
    registry = build_default_tool_registry(
        index, WebResearchAgent(index), workspace_root=tmp_path, multimodal_service=service,
    )

    result = registry.call("knowledge.search_directory_images", directory=str(directory), query="red dog")

    assert result.ok
    assert result.output["matches"][0]["path"] == str((directory / "red.jpg").resolve())


def test_router_can_select_directory_visual_search_without_planner(tmp_path: Path) -> None:
    directory = tmp_path / "dogs"
    directory.mkdir()
    registry = ToolRegistry()
    registry.register(ToolSpec(
        name="knowledge.search_directory_images", category="knowledge", description="Find local images",
        parameters=[ToolParameter("directory", "string"), ToolParameter("query", "string")],
        handler=lambda directory, query: {},
    ))

    class Client:
        def chat(self, *args, **kwargs):
            return json.dumps({
                "mode": "tool_call", "reason": "single read-only search", "direct_response": "",
                "tool_name": "knowledge.search_directory_images",
                "arguments": {"directory": str(directory), "query": "a dog wearing red clothes"},
                "confidence": 0.95,
            })

    decision = IntentRouter(Client()).route(
        f'在"{directory}"这个目录下帮我找到一张穿红衣服的狗的图片', registry,
    )
    assert decision.mode == "tool_call"
    assert decision.tool_name == "knowledge.search_directory_images"
    assert decision.arguments["directory"] == str(directory)


def test_planned_directory_image_search_executes_instead_of_fallback(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setenv("ALLOW_LOCAL_FALLBACK", "true")
    index = DocumentIndex(tmp_path / "index" / "index.json")
    agent = DocumentQAAgent(index)
    directory = tmp_path / "dogs"
    directory.mkdir()
    plan = {
        "route": "single_agent", "score": 3, "reason": "", "valid": True, "error": "",
        "steps": [{"id": "knowledge", "agent": "knowledge", "depends_on": [],
                   "allowed_tools": ["knowledge.search_directory_images"],
                   "requires_human": False,
                   "arguments": {"directory": str(directory), "query": "red dog"}}],
    }
    monkeypatch.setattr(agent.intent_router, "route", lambda *args, **kwargs: IntentDecision(
        mode="plan_task", reason="Directory visual search", required_tools=["knowledge.search_directory_images"],
    ))
    monkeypatch.setattr(agent, "plan_task", lambda *args, **kwargs: plan)
    monkeypatch.setattr(agent.plan_repair, "repair", lambda **kwargs: PlanRepairResult(plan))
    registry = ToolRegistry()
    registry.register(ToolSpec(
        name="knowledge.search_directory_images", category="knowledge", description="Read-only search",
        parameters=[ToolParameter("directory", "string"), ToolParameter("query", "string")],
        handler=lambda directory, query: {"directory": directory, "query": query, "scanned": 1,
                                          "skipped": 0, "matches": [{"path": str(Path(directory) / "red.jpg"),
                                                                       "score": 0.9}]},
    ))
    agent.tool_registry = registry
    result = agent.answer(f'在"{directory}"这个目录下帮我找到一张穿红衣服的狗的图片')

    assert "red.jpg" in result.answer
    assert "没有找到受支持" not in result.answer
    assert any(step.name == "execute_plan_node" for step in result.steps)
    assert any(step.name == "execute_tool" for step in result.steps)


def test_planned_directory_search_uses_quoted_path_when_plan_omits_it() -> None:
    plan = {"steps": [{"allowed_tools": ["knowledge.search_directory_images"],
                       "arguments": {"query": "a dog wearing red clothes"}}]}
    question = '在"C:\\Users\\DELL\\Desktop\\dataset\\dog"这个目录下找一张穿红衣服的狗的图片'
    result = DocumentQAAgent._planned_directory_image_search(question, plan, IntentDecision(mode="plan_task"))
    assert result is not None
    assert result["directory"] == r"C:\Users\DELL\Desktop\dataset\dog"
    assert result["query"] == "a dog wearing red clothes"
