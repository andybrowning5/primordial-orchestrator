"""Primordial Orchestrator AgentStore adapter.

Speaks the Primordial Protocol (NDJSON over stdin/stdout).
"""

from __future__ import annotations

import json
import sys
from typing import Any


def send(msg: dict) -> None:
    """Write a Primordial Protocol message to stdout."""
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def _extract_response(result: dict[str, Any]) -> str:
    """Extract assistant text from a LangGraph result."""
    if "messages" not in result:
        return ""
    for msg in reversed(result["messages"]):
        if getattr(msg, "type", None) == "ai" and getattr(msg, "content", None):
            content = msg.content
            if isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        parts.append(block)
                content = "\n".join(parts)
            if content:
                return content
    return ""


__emitted_tool_calls: set[str] = set()


def handle_message(agent: Any, config: dict, content: str, message_id: str) -> None:
    """Process a user message through the orchestrator agent."""
    try:
        messages: list[dict[str, str]] = [{"role": "user", "content": content}]
        final_response = ""

        if hasattr(agent, "stream"):
            for event in agent.stream(
                {"messages": messages},
                config=config,
                stream_mode="values",
            ):
                if not isinstance(event, dict) or "messages" not in event:
                    continue

                for msg in event["messages"]:
                    if (
                        getattr(msg, "type", None) == "ai"
                        and hasattr(msg, "tool_calls")
                        and msg.tool_calls
                    ):
                        for tc in msg.tool_calls:
                            tc_id = tc.get("id") or tc.get("name", "")
                            if tc_id in _emitted_tool_calls:
                                continue
                            _emitted_tool_calls.add(tc_id)
                            tool_name = tc.get("name", "unknown")
                            tool_args = tc.get("args", {})
                            query = (
                                tool_args.get("query")
                                or tool_args.get("message")
                                or ""
                            )
                            desc = f"{tool_name}({query})" if query else tool_name
                            send({
                                "type": "activity",
                                "tool": tool_name,
                                "description": desc,
                                "message_id": message_id,
                            })

                for msg in reversed(event["messages"]):
                    if getattr(msg, "type", None) != "ai":
                        continue
                    msg_content = getattr(msg, "content", "")
                    if isinstance(msg_content, str) and msg_content.strip():
                        final_response = msg_content
                        break
                    elif isinstance(msg_content, list):
                        for block in msg_content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                final_response = block.get("text", "")
                                break
                        if final_response:
                            break
        else:
            result = agent.invoke({"messages": messages}, config=config)
            final_response = _extract_response(result)

        if final_response:
            send({
                "type": "response",
                "content": final_response,
                "message_id": message_id,
                "done": True,
            })
        else:
            send({
                "type": "response",
                "content": "Processed your message but got no text response.",
                "message_id": message_id,
                "done": True,
            })

    except Exception as exc:
        send({"type": "error", "error": str(exc), "message_id": message_id})


def main() -> None:
    """Primordial Protocol main loop."""
    from src.agent import create_orchestrator_agent

    agent = create_orchestrator_agent()
    config = {"configurable": {"thread_id": "primordial-orchestrator-default"}}

    send({"type": "ready"})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue

        if msg["type"] == "shutdown":
            break
        if msg["type"] == "message":
            handle_message(agent, config, msg["content"], msg["message_id"])


if __name__ == "__main__":
    main()
