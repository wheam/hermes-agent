"""Configurable personal-context response guard.

The plugin is intentionally provider- and domain-agnostic. It becomes active
only when ``$HERMES_HOME/personal-context-guard.json`` names one or more read
tools. When a user refers to their own prior/stable context and the current
turn has not called any configured read tool, the plugin withholds the
candidate answer and asks the model to fetch evidence first.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set

logger = logging.getLogger(__name__)

_DEFAULT_MARKERS = (
    "我的",
    "我那",
    "我这台",
    "我之前",
    "我上次",
    "我们之前",
    "咱们之前",
    "上次说的",
    "之前说的",
    "还记得",
    "你记得",
)
_ENGLISH_PERSONAL_REFERENCE = re.compile(
    r"\b(?:my|mine|our|ours)\b|\b(?:last time|previously|we discussed|you remember)\b",
    re.IGNORECASE,
)
_DEFAULT_MAX_ATTEMPTS = 2


def _config_path() -> Path:
    root = Path(os.getenv("HERMES_HOME") or (Path.home() / ".hermes"))
    return root / "personal-context-guard.json"


def _load_config() -> Optional[Dict[str, Any]]:
    path = _config_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception as exc:
        logger.warning("personal-context guard config unreadable: %s", exc)
        return None
    if not isinstance(data, dict) or data.get("enabled", True) is False:
        return None
    read_tools = data.get("read_tools")
    if not isinstance(read_tools, list) or not any(
        isinstance(name, str) and name.strip() for name in read_tools
    ):
        return None
    return data


def _message_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""
    parts = []
    for item in value:
        if isinstance(item, dict) and item.get("type") in {"text", "input_text"}:
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts)


def _has_personal_reference(text: str, markers: Iterable[str]) -> bool:
    if any(marker and marker in text for marker in markers):
        return True
    return bool(_ENGLISH_PERSONAL_REFERENCE.search(text))


def _tool_call_name(tool_call: Any) -> str:
    if isinstance(tool_call, dict):
        function = tool_call.get("function")
        if isinstance(function, dict):
            name = function.get("name")
            return name if isinstance(name, str) else ""
        name = tool_call.get("name")
        return name if isinstance(name, str) else ""
    function = getattr(tool_call, "function", None)
    name = getattr(function, "name", None)
    return name if isinstance(name, str) else ""


def _called_tools(conversation_history: Iterable[Any]) -> Set[str]:
    called: Set[str] = set()
    for message in conversation_history:
        if not isinstance(message, dict):
            continue
        for tool_call in message.get("tool_calls") or []:
            name = _tool_call_name(tool_call)
            if name:
                called.add(name)
        if message.get("role") == "tool":
            name = message.get("name")
            if isinstance(name, str) and name:
                called.add(name)
    return called


def _on_pre_response(
    *,
    user_message: Any = "",
    conversation_history: Optional[list] = None,
    available_tools: Optional[list] = None,
    attempt: int = 0,
    **_: Any,
) -> Optional[Dict[str, str]]:
    config = _load_config()
    if config is None:
        return None

    try:
        max_attempts = max(1, min(int(config.get("max_attempts", _DEFAULT_MAX_ATTEMPTS)), 2))
    except (TypeError, ValueError):
        max_attempts = _DEFAULT_MAX_ATTEMPTS
    if attempt >= max_attempts:
        return None

    text = _message_text(user_message)
    configured_markers = config.get("personal_markers")
    markers = (
        tuple(marker for marker in configured_markers if isinstance(marker, str))
        if isinstance(configured_markers, list)
        else _DEFAULT_MARKERS
    )
    if not text or not _has_personal_reference(text, markers):
        return None

    read_tools = {
        name.strip()
        for name in config.get("read_tools", [])
        if isinstance(name, str) and name.strip()
    }
    available = {
        name for name in (available_tools or []) if isinstance(name, str)
    }
    usable = sorted(read_tools & available)
    if not usable:
        return None
    if _called_tools(conversation_history or []) & read_tools:
        return None

    preferred_tool = usable[0]
    return {
        "action": "continue",
        "message": (
            "[System: The user's request refers to personal or previously stored "
            "context, but this turn has not read the configured personal knowledge "
            f"source. Before answering, call `{preferred_tool}` with a concise query "
            "covering the referenced entity and the current decision. Treat terminal, "
            "local-file, session-search, and system information as belonging only to "
            "their explicit host/session; never substitute them for another person, "
            "device, account, place, or project. If the knowledge source reports a "
            "gap, state the exact missing facts and ask only for the minimum needed "
            "to answer. Do not guess.]"
        ),
        "fallback_response": (
            "我还没能从个人知识库核实这项信息，因此先不猜。"
            "请告诉我完成当前判断所需的最小事实；稳定信息可在你同意后保存。"
        ),
    }


def register(ctx) -> None:
    ctx.register_hook("pre_response", _on_pre_response)
