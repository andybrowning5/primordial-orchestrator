"""Primordial Orchestrator: discovers and delegates to agents dynamically."""

import json
import os
import sys
from typing import Any

from deepagents import create_deep_agent
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()


def _emit(msg: dict) -> None:
    """Emit a Primordial Protocol event to stdout."""
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()

SYSTEM_PROMPT = """\
You are the Primordial Orchestrator. You discover and delegate to specialized \
agents on the Primordial AgentStore.

## Workflow

1. When the user asks something you can't answer directly, call \
**search_agents** with a relevant query to find agents that can help.
2. Review the results — read each agent's name and description to pick the \
best match.
3. Call **start_agent** with the chosen agent's URL to spawn it.
4. Call **message_agent** with the session ID and the user's request.
5. You can have multi-turn conversations — send follow-up messages to the \
same session.
6. Use **monitor_agent** to see what the sub-agent has been doing.
7. Call **stop_agent** when done with a sub-agent.

## Rules

- Always search before delegating — don't guess agent URLs.
- If no agent matches, tell the user honestly.
- For simple greetings or clarifications, respond directly without delegating.
- If a task spans multiple domains, start multiple sub-agents.
- Always tell the user which agent you're delegating to and why.
- Use monitor_agent to check on sub-agent progress if a response seems \
incomplete.
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
    from primordial_delegate import search
    return json.dumps(search(query))


@tool
def list_all_agents() -> str:
    """List all agents on the Primordial AgentStore sorted by popularity."""
    print("[orchestrator] listing all agents", file=sys.stderr)
    from primordial_delegate import search_all
    return json.dumps(search_all())


@tool
def start_agent(agent_url: str) -> str:
    """Spawn a sub-agent for multi-turn conversation.

    Returns a session_id to use with message_agent and monitor_agent.

    Args:
        agent_url: GitHub URL of the agent to run.
    """
    print(f"[orchestrator] starting: {agent_url}", file=sys.stderr)
    from primordial_delegate import run_agent
    return run_agent(agent_url)


@tool
def message_agent(session_id: str, message: str) -> str:
    """Send a message to a running sub-agent and get its response.

    Returns the response text and a summary of tools the sub-agent used.

    Args:
        session_id: Session ID from start_agent.
        message: The message to send.
    """
    print(f"[orchestrator] messaging {session_id}: {message}", file=sys.stderr)
    from primordial_delegate import message_agent_stream

    activities = []
    final_response = ""

    for event in message_agent_stream(session_id, message):
        if event.get("type") == "stream_event":
            inner = event.get("event", {})
            if inner.get("type") == "activity":
                tool_name = inner.get("tool", "")
                desc = inner.get("description", "")
                activities.append({"tool": tool_name, "description": desc})
                # Emit to stdout so the TUI shows sub-agent progress
                _emit({
                    "type": "activity",
                    "tool": f"sub:{tool_name}",
                    "description": desc,
                })
            elif inner.get("type") == "response" and inner.get("done"):
                final_response = inner.get("content", "")
                # Show a preview of the sub-agent's response
                preview = final_response.replace("\n", " ")[:150].strip()
                if len(final_response) > 150:
                    preview += "..."
                _emit({
                    "type": "activity",
                    "tool": "sub:response",
                    "description": preview,
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
    from primordial_delegate import monitor_agent as _mon
    lines = _mon(session_id)
    return "\n".join(lines) if lines else "No output yet."


@tool
def stop_agent(session_id: str) -> str:
    """Shutdown a sub-agent session and save its state.

    Args:
        session_id: Session ID from start_agent.
    """
    print(f"[orchestrator] stopping {session_id}", file=sys.stderr)
    from primordial_delegate import stop_agent as _stop
    _stop(session_id)
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
