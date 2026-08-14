"""Behavior tests for the opt-in personal-context response guard."""

import importlib
import json


def _plugin(tmp_path, monkeypatch, config):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "personal-context-guard.json").write_text(
        json.dumps(config), encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    module = importlib.import_module("plugins.personal_context_guard")
    return importlib.reload(module)


def _available():
    return [
        "mcp__substrate_kb__recall",
        "mcp__substrate_kb__search",
        "session_search",
    ]


def _config():
    return {
        "read_tools": [
            "mcp__substrate_kb__recall",
            "mcp__substrate_kb__search",
        ],
        "max_attempts": 2,
    }


def test_personal_reference_without_read_is_held(tmp_path, monkeypatch):
    plugin = _plugin(tmp_path, monkeypatch, _config())
    directive = plugin._on_pre_response(
        user_message="我的另一台 Windows 笔记本有多少内存？",
        conversation_history=[{"role": "user", "content": "same"}],
        available_tools=_available(),
    )
    assert directive["action"] == "continue"
    assert "mcp__substrate_kb__recall" in directive["message"]
    assert "不猜" in directive["fallback_response"]


def test_prior_session_search_does_not_satisfy_guard(tmp_path, monkeypatch):
    plugin = _plugin(tmp_path, monkeypatch, _config())
    directive = plugin._on_pre_response(
        user_message="What did my other laptop have?",
        conversation_history=[{
            "role": "assistant",
            "tool_calls": [{"function": {"name": "session_search"}}],
        }],
        available_tools=_available(),
    )
    assert directive is not None


def test_configured_read_satisfies_guard(tmp_path, monkeypatch):
    plugin = _plugin(tmp_path, monkeypatch, _config())
    directive = plugin._on_pre_response(
        user_message="我的另一台 Windows 笔记本有多少内存？",
        conversation_history=[{
            "role": "assistant",
            "tool_calls": [{
                "function": {"name": "mcp__substrate_kb__recall"},
            }],
        }],
        available_tools=_available(),
    )
    assert directive is None


def test_non_personal_question_is_not_held(tmp_path, monkeypatch):
    plugin = _plugin(tmp_path, monkeypatch, _config())
    assert plugin._on_pre_response(
        user_message="Windows 11 最低需要多少内存？",
        conversation_history=[],
        available_tools=_available(),
    ) is None


def test_missing_or_unavailable_configuration_is_inert(tmp_path, monkeypatch):
    plugin = _plugin(tmp_path, monkeypatch, {"read_tools": ["missing_tool"]})
    assert plugin._on_pre_response(
        user_message="我的电脑有多少内存？",
        conversation_history=[],
        available_tools=_available(),
    ) is None
