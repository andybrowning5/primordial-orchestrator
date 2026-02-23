"""Primordial Orchestrator: discovers and delegates to agents dynamically."""

import json
import os
import socket
import sys
from typing import Any, Generator

from deepagents import create_deep_agent
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# ---------------------------------------------------------------------------
# Raw NDJSON delegation helpers (connects to /tmp/_primordial_delegate.sock)
# ---------------------------------------------------------------------------

_SOCK_PATH = "/tmp/_primordial_delegate.sock"
_sock: socket.socket | None = None
_sock_buf: bytes = b""


def _get_sock() -> socket.socket:
    global _sock
    if _sock is None:
        _sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        _sock.connect(_SOCK_PATH)
    return _sock


def _send(obj: dict) -> None:
    _get_sock().sendall((json.dumps(obj) + "\n").encode())


def _read_line() -> dict:
    global _sock_buf
    sock = _get_sock()
    while b"\n" not in _sock_buf:
        chunk = sock.recv(8192)
        if not chunk:
            raise ConnectionError("Delegation socket closed")
        _sock_buf += chunk
    line, _sock_buf = _sock_buf.split(b"\n", 1)
    return json.loads(line)


def _read_stream(stop_fn) -> Generator[dict, None, None]:
    while True:
        msg = _read_line()
        yield msg
        if stop_fn(msg):
            return


def _emit(msg: dict) -> None:
    """Emit a Primordial Protocol event to stdout."""
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()

SYSTEM_PROMPT = """\
You are the Primordial Orchestrator. You coordinate specialized agents on \
the Primordial AgentStore.

## Core Principle

You are a coordinator, not a doer. For every user request, call \
**list_all_agents** first to see what's available. If any agent is better \
suited for the task, delegate to it. Only respond directly if no agent fits \
or it's a simple greeting/clarification.

## Workflow

1. User sends a request → call **list_all_agents** to see available agents.
2. Pick the best agent for the task based on its description.
3. Call **start_agent** with the chosen agent's URL to spawn it.
4. Call **message_agent** with the session ID and the user's request.
5. You can have multi-turn conversations — send follow-up messages to the \
same session.
6. Use **monitor_agent** to check on sub-agent progress.
7. Call **stop_agent** when done with a sub-agent.

## Rules

- Always list agents before responding to a task — don't guess or skip this.
- If no agent matches, tell the user and attempt it yourself.
- If a task spans multiple domains, start multiple sub-agents.
- Tell the user which agent you're delegating to and why.
"""


@tool
def search_agents(query: str) -> str:
    """Semantic search for agents on the Primordial AgentStore.

    Returns the top 5 agents ranked by relevance to your query.

    Args:
        query: Natural language description of the capability needed
            (e.g., "web research", "task management", "code review").
    """
    print(f"[orchestrator] searching: {query}", file=sys.stderr)
    _send({"type": "search", "query": query})
    result = _read_line()
    return json.dumps(result.get("agents", []))


@tool
def list_all_agents() -> str:
    """List all agents on the Primordial AgentStore sorted by popularity."""
    print("[orchestrator] listing all agents", file=sys.stderr)
    _send({"type": "search_all"})
    result = _read_line()
    return json.dumps(result.get("agents", []))


@tool
def start_agent(agent_url: str) -> str:
    """Spawn a sub-agent for multi-turn conversation.

    Returns a session_id to use with message_agent and monitor_agent.

    Args:
        agent_url: GitHub URL of the agent to run.
    """
    print(f"[orchestrator] starting: {agent_url}", file=sys.stderr)
    _send({"type": "run", "agent_url": agent_url})
    for msg in _read_stream(lambda m: m["type"] != "setup_status"):
        if msg["type"] == "setup_status":
            _emit({
                "type": "activity",
                "tool": "sub:setup",
                "description": msg.get("status", ""),
                "session_id": msg.get("session_id", ""),
            })
        elif msg["type"] == "session":
            return msg["session_id"]
        elif msg["type"] == "error":
            return f"Error: {msg.get('error', 'unknown')}"
    return "Error: unexpected end of stream"


@tool
def message_agent(session_id: str, message: str) -> str:
    """Send a message to a running sub-agent and get its response.

    Returns the response text and a summary of tools the sub-agent used.

    Args:
        session_id: Session ID from start_agent.
        message: The message to send.
    """
    print(f"[orchestrator] messaging {session_id}: {message}", file=sys.stderr)
    _send({"type": "message", "session_id": session_id, "content": message})

    activities = []
    final_response = ""

    for event in _read_stream(lambda m: m.get("done", False)):
        if event.get("type") != "stream_event":
            continue
        inner = event.get("event", {})
        if inner.get("type") == "activity":
            tool_name = inner.get("tool", "")
            desc = inner.get("description", "")
            activities.append({"tool": tool_name, "description": desc})
            args_desc = desc
            if desc.startswith(f"{tool_name}(") and desc.endswith(")"):
                args_desc = desc[len(tool_name) + 1:-1]
            _emit({
                "type": "activity",
                "tool": f"sub:{tool_name}",
                "description": args_desc,
                "session_id": session_id,
            })
        elif inner.get("type") == "response" and inner.get("done"):
            final_response = inner.get("content", "")
            preview = final_response.replace("\n", " ")[:150].strip()
            if len(final_response) > 150:
                preview += "..."
            _emit({
                "type": "activity",
                "tool": "sub:response",
                "description": preview,
                "session_id": session_id,
            })

    return json.dumps({"response": final_response, "activities": activities})


@tool
def monitor_agent(session_id: str) -> str:
    """View the last 1000 lines of a sub-agent's output.

    Shows tool calls, searches, responses, and errors — like scrolling
    through a terminal to see what the sub-agent has been doing.

    Args:
        session_id: Session ID from start_agent.
    """
    _send({"type": "monitor", "session_id": session_id})
    result = _read_line()
    lines = result.get("lines", [])
    return "\n".join(lines) if lines else "No output yet."


@tool
def stop_agent(session_id: str) -> str:
    """Shutdown a sub-agent session and save its state.

    Args:
        session_id: Session ID from start_agent.
    """
    print(f"[orchestrator] stopping {session_id}", file=sys.stderr)
    _send({"type": "stop", "session_id": session_id})
    _read_line()
    return "Agent stopped."


def get_model(model_name: str | None = None) -> Any:
    """Initialize the chat model."""
    if model_name:
        return init_chat_model(model_name)
    if os.getenv("ANTHROPIC_API_KEY"):
        return init_chat_model("anthropic:claude-sonnet-4-5-20250929")
    raise ValueError("No ANTHROPIC_API_KEY found.")


def create_orchestrator_agent(model_name: str | None = None) -> Any:
    """Create the primordial orchestrator deep agent."""
    model = get_model(model_name)

    return create_deep_agent(
        model=model,
        tools=[
            search_agents,
            list_all_agents,
            start_agent,
            message_agent,
            monitor_agent,
            stop_agent,
        ],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=MemorySaver(),
    )
