from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolate_external_services(monkeypatch: pytest.MonkeyPatch) -> None:
    """默认阻断本机 .env 凭证进入确定性单元测试；需要 API 的测试可自行覆盖。"""
    for name in ("DASHSCOPE_API_KEY", "LLM_API_KEY", "EMBEDDING_API_KEY"):
        monkeypatch.setenv(name, "")
    monkeypatch.setenv("ALLOW_LOCAL_FALLBACK", "true")


@pytest.fixture
def base(tmp_path: Path) -> Path:
    """兼容历史测试使用的隔离工作目录参数。

    旧测试通过模块内的 ``main()`` 手工创建临时目录；统一使用 pytest 后，
    由内置 ``tmp_path`` 保证每条用例互不污染。
    """
    return tmp_path
