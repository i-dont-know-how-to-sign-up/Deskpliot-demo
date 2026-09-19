from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
INDEX_DIR = DATA_DIR / "index"
WORKSPACE_DIR = DATA_DIR / "workspace"
REPORTS_DIR = WORKSPACE_DIR / "reports"
LOG_DIR = DATA_DIR / "logs"
MEMORY_DIR = DATA_DIR / "memory"
MEMORY_WORKSPACE_DIR = MEMORY_DIR / "workspace"
MEMORY_SESSIONS_DIR = MEMORY_DIR / "sessions"
MEMORY_INDEX_DIR = MEMORY_DIR / "indexes"
MEMORY_VECTOR_DIR = MEMORY_DIR / "vectors"


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _env_value(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip() != "":
            return value
    return default


@dataclass(frozen=True)
class ModelConfig:
    llm_api_key: str
    llm_base_url: str
    llm_model: str
    embedding_api_key: str
    embedding_base_url: str
    embedding_model: str
    qwen_enable_thinking: bool | None
    allow_local_fallback: bool
    search_provider: str
    search_api_key: str
    search_endpoint: str
    research_max_results: int
    browser_channel: str
    browser_headless: bool
    browser_timeout_ms: int
    search_engine: str


def ensure_dirs() -> None:
    for path in (
        DATA_DIR,
        INDEX_DIR,
        WORKSPACE_DIR,
        REPORTS_DIR,
        LOG_DIR,
        MEMORY_DIR,
        MEMORY_WORKSPACE_DIR,
        MEMORY_SESSIONS_DIR,
        MEMORY_INDEX_DIR,
        MEMORY_VECTOR_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def load_config() -> ModelConfig:
    _load_dotenv(ROOT_DIR / ".env")
    allow_fallback = _env_value("ALLOW_LOCAL_FALLBACK", default="true").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    dashscope_api_key = _env_value("DASHSCOPE_API_KEY")
    dashscope_base_url = _env_value("DASHSCOPE_BASE_URL", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    qwen_thinking_raw = os.getenv("QWEN_ENABLE_THINKING")
    qwen_enable_thinking = None
    if qwen_thinking_raw is not None:
        qwen_enable_thinking = qwen_thinking_raw.lower() in {"1", "true", "yes", "on"}
    try:
        research_max_results = int(_env_value("RESEARCH_MAX_RESULTS", default="5"))
    except ValueError:
        research_max_results = 5
    try:
        browser_timeout_ms = int(_env_value("BROWSER_TIMEOUT_MS", default="30000"))
    except ValueError:
        browser_timeout_ms = 30000
    browser_headless = _env_value("BROWSER_HEADLESS", default="true").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    return ModelConfig(
        llm_api_key=_env_value("DASHSCOPE_API_KEY", "LLM_API_KEY"),
        llm_base_url=_env_value("DASHSCOPE_BASE_URL", "LLM_BASE_URL", default=dashscope_base_url),
        llm_model=_env_value("QWEN_MODEL", "LLM_MODEL", default="qwen3.7-plus"),
        embedding_api_key=_env_value("DASHSCOPE_API_KEY", "EMBEDDING_API_KEY", "LLM_API_KEY"),
        embedding_base_url=_env_value("DASHSCOPE_BASE_URL", "EMBEDDING_BASE_URL", "LLM_BASE_URL", default=dashscope_base_url),
        embedding_model=_env_value("QWEN_EMBEDDING_MODEL", "EMBEDDING_MODEL", default="text-embedding-v4"),
        qwen_enable_thinking=qwen_enable_thinking,
        allow_local_fallback=allow_fallback,
        search_provider=_env_value("SEARCH_PROVIDER", default="duckduckgo").lower(),
        search_api_key=_env_value("SEARCH_API_KEY", "BING_SEARCH_API_KEY", "TAVILY_API_KEY"),
        search_endpoint=_env_value("SEARCH_ENDPOINT"),
        research_max_results=max(1, min(research_max_results, 10)),
        browser_channel=_env_value("BROWSER_CHANNEL", default="chrome").lower(),
        browser_headless=browser_headless,
        browser_timeout_ms=max(5000, min(browser_timeout_ms, 120000)),
        search_engine=_env_value("SEARCH_ENGINE", default="bing").lower(),
    )
