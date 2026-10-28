from __future__ import annotations

import io
import json
import urllib.error
import urllib.request

import pytest

from deskpilot.decision.jev import JevClient, JevConfig


def test_netlify_config_uses_separate_endpoint_and_token(monkeypatch) -> None:
    monkeypatch.setenv("JEV_PROVIDER", "netlify")
    monkeypatch.setenv("JEV_MODE", "shadow")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-secret")
    monkeypatch.setenv("JEV_ENDPOINT", "https://openrouter.ai/api/alpha/decisions")
    monkeypatch.setenv("NETLIFY_JEV_TOKEN", "bridge-secret")
    monkeypatch.setenv("NETLIFY_JEV_ENDPOINT", "https://sample.netlify.app/api/jev")
    config = JevConfig.from_env()
    assert config.provider == "netlify"
    assert config.api_key == "bridge-secret"
    assert config.endpoint == "https://sample.netlify.app/api/jev"
    assert config.credential_name == "NETLIFY_JEV_TOKEN"
    assert config.model == "netlify/systemOne"


def test_openrouter_remains_default(monkeypatch) -> None:
    monkeypatch.delenv("JEV_PROVIDER", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-secret")
    config = JevConfig.from_env()
    assert config.provider == "openrouter"
    assert config.api_key == "openrouter-secret"
    assert config.credential_name == "OPENROUTER_API_KEY"


def test_netlify_posts_typed_decision_without_exposing_content_in_usage(monkeypatch) -> None:
    config = JevConfig(provider="netlify", api_key="bridge-secret",
                       endpoint="https://sample.netlify.app/api/jev")
    client = JevClient(config)
    requests = []

    def fake_urlopen(request, timeout):
        requests.append((request, timeout))
        return io.BytesIO(json.dumps({
            "answers": {"flag": {"type": "noul", "noul": 0.9}},
            "usage": {"input_tokens": 7, "output_tokens": 2},
        }).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    answers = client.judge("private input", {"flag": {"type": "noul", "instructions": "yes?"}})
    request, timeout = requests[0]
    assert request.full_url == config.endpoint
    assert request.get_header("Authorization") == "Bearer bridge-secret"
    assert json.loads(request.data)["state"] == "private input"
    assert "model" not in json.loads(request.data)
    assert answers["flag"]["noul"] == 0.9
    assert client.token_usage["total_tokens"] == 9
    assert "private input" not in str(client.usage_events)


def test_netlify_http_error_is_named_without_leaking_response(monkeypatch) -> None:
    client = JevClient(JevConfig(provider="netlify", api_key="token",
                                 endpoint="https://sample.netlify.app/api/jev"))

    def denied(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 403, "denied", {}, io.BytesIO(b"private data"))

    monkeypatch.setattr(urllib.request, "urlopen", denied)
    with pytest.raises(RuntimeError, match="Netlify Jev HTTP 403") as error:
        client.judge("private input", {"flag": {"type": "noul"}})
    assert "private data" not in str(error.value)


def test_netlify_requires_https_and_own_token() -> None:
    client = JevClient(JevConfig(provider="netlify", endpoint="https://site.netlify.app/api/jev"))
    with pytest.raises(RuntimeError, match="NETLIFY_JEV_TOKEN"):
        client.judge("a", {"flag": {"type": "noul"}})
    client = JevClient(JevConfig(provider="netlify", api_key="token", endpoint="http://localhost/api/jev"))
    with pytest.raises(ValueError, match="NETLIFY_JEV_ENDPOINT must use HTTPS"):
        client.judge("a", {"flag": {"type": "noul"}})
