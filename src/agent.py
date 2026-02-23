"""Primordial Orchestrator: discovers and delegates to agents dynamically."""

import json
import os
import subprocess
import sys
from typing import Any

from deepagents import create_deep_agent
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

load_dotenv()

SYSTEM_PROMPT = """\
You are the Primordial Orchestrator. You do not have built-in knowledge or \
capabilities beyond conversation. Instead, you discover and delegate to \
specialized agents on the Primordial AgentStore.

## Workflow

1. When the user asks something you can't answer directly, call \
**search_agents** with a relevant query to find agents that can help.
2. Review the results — read each agent's name and description to pick the \
best match.
3. Call **run_agent** with the chosen agent's URL and the user's request.
4. Synthesize the sub-agent's response into a clear answer for the user.

## Rules

- Always search before delegating — don't guess agent URLs.
- If no agent matches, tell the user honestly and suggest they try a \
different query.
- For simple greetings or clarifications, respond directly without delegating.
- If a task spans multiple domains (e.g., research + task management), search \
for and delegate to multiple agents sequentially.
- Always tell the user which agent you're delegating to and why.
"""


@tool
def search_agents(query: str) -> str:
    """Search the Primordial AgentStore for agents matching a query.

    Returns a JSON list of available agents with name, description, and URL.
    Use this to discover which agents can help with the user's request.

    Args:
        query: Search terms describing the capability needed
            (e.g., "web research", "task management", "code review").
    """
    print(f"[orchestrator] searching agents: {query}", file=sys.stderr)
    try:
        result = subprocess.run(
            ["primordial", "search", query, "--json"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            return f"Search failed: {result.stderr.strip()}"
        return result.stdout.strip()
    except FileNotFoundError:
        return "Error: 'primordial' CLI not found. Is it installed?"
    except subprocess.TimeoutExpired:
        return "Search timed out."


@tool
def run_agent(agent_url: str, message: str) -> str:
    """Delegate a task to a Primordial agent by its GitHub URL.

    Spawns the agent as a sub-agent, sends it a message, and returns its
    response. Use search_agents first to find the right agent URL.

    Args:
        agent_url: The GitHub URL of the agent to run
            (e.g., "https://github.com/user/agent-name").
        message: The task or question to send to the agent.
    """
    print(f"[orchestrator] delegating to {agent_url}: {message}", file=sys.stderr)
    try:
        proc = subprocess.Popen(
            ["primordial", "run", agent_url, "--agent-read", "--yes"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Wait for ready
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            msg = json.loads(line)
            if msg.get("type") == "ready":
                break

        # Send message
        proc.stdin.write(json.dumps({
            "type": "message",
            "content": message,
            "message_id": "delegate-1",
        }) + "\n")
        proc.stdin.flush()

        # Collect response
        response = ""
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            msg = json.loads(line)
            if msg.get("type") == "response":
                response = msg.get("content", "")
                if msg.get("done"):
                    break
            elif msg.get("type") == "error":
                response = f"Sub-agent error: {msg.get('error', 'unknown')}"
                break

        # Shutdown
        proc.stdin.write(json.dumps({"type": "shutdown"}) + "\n")
        proc.stdin.flush()
        proc.wait(timeout=10)

        return response or "Sub-agent returned no response."

    except FileNotFoundError:
        return "Error: 'primordial' CLI not found. Is it installed?"
    except Exception as e:
        return f"Delegation error: {e}"


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
        tools=[search_agents, run_agent],
        system_prompt=SYSTEM_PROMPT,
    )
