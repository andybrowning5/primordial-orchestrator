# Primordial Orchestrator

Dynamic orchestrator that discovers and delegates to any agent on the Primordial AgentStore at runtime. No hardcoded agent URLs — it searches the store, picks the best agent for the job, and delegates.

## How It Works

1. You ask a question or give a task
2. The orchestrator runs `primordial search` to find relevant agents
3. It picks the best match based on the agent's description
4. It spawns the agent via `primordial run` and delegates your request
5. It synthesizes the response back to you

## Required API Keys

- `ANTHROPIC_API_KEY` — for the orchestrator's LLM reasoning

Sub-agents manage their own keys.

## Usage

```bash
primordial run https://github.com/andybrowning5/primordial-orchestrator
```

## Example

```
You: What are the latest trends in AI agents?
Orchestrator: [searches for research agents → finds web-research-agent → delegates → returns sourced briefing]

You: Help me organize my project tasks
Orchestrator: [searches for task agents → finds cadence → delegates → returns prioritized task list]
```
