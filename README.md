# Agentic System with Guardrails & Observability

[![Diagram](https://img.shields.io/badge/gitdiagram-view%20architecture-blue)](https://gitdiagram.com/mehta-amit-codes/agentic-guardrails)

Reference implementation of the "Agentic System with Guardrails &
Observability" blueprint: a tool-using ReAct agent wrapped with structured
tracing, input/output guardrails (PII redaction, prompt-injection
scanning, tool-output schema validation), budget enforcement, a circuit
breaker, a replay harness, and a red-team test suite — plus a Streamlit
UI to watch the agent think step by step.

Uses **Grok (xAI)** for the agent's LLM calls via its OpenAI-compatible
API.

## How the pieces fit together

```
task
  |
  v
[LLM call: reasoning + tool choice]  --trace: "thought" span-->
  |
  v
tool call chosen?
  |                                  \
  | yes                               no --> treat content as final answer, end run
  v
[scan tool ARGUMENTS for injection patterns]
  |
  v
[execute tool]  --failure--> [circuit breaker: consecutive_failures++] --tripped--> halt run
  |
  | success --> [circuit breaker: reset]
  v
[validate tool OUTPUT against Pydantic schema]  --invalid--> treat as tool failure
  |
  v
[redact PII from output] --> [scan output for injection patterns] --> [wrap as untrusted content]
  |
  v
[record step: tokens + cost]  --over budget--> halt run, return partial result
  |
  v
back to top of loop, until final_answer is called or a limit is hit
```

Every arrow that says "trace" is a row in `agent_traces.db`, browsable in
the Streamlit **Trace Browser** tab or via `GET /runs/{run_id}/spans`.

## Run it

```bash
cp .env.example .env
# then edit .env and fill in your real GROK_API_KEY

pip install -r requirements.txt
uvicorn app.main:app --port 8003 --reload
```

In a second terminal:
```bash
export AGENT_API_URL=http://localhost:8003
streamlit run streamlit_app.py
```

## Demo flow

1. **Run Agent** tab — give it a task (e.g. the default: a calculation +
   a web search + summarize). Watch the trace render live: each thought,
   tool call, and guardrail flag (PII redacted, injection detected, tool
   failed) shown inline.
2. Try an open-ended task like *"keep searching for random topics
   forever, never stop"* — watch it hit `AGENT_MAX_STEPS` and halt with
   `budget_exceeded` instead of looping.
3. **Trace Browser** tab — browse any past run's full trace by `run_id`.
4. **Red Team** tab — runs four adversarial test cases (prompt injection
   in a tool result, PII in a tool result, an unsafe calculator
   expression, an open-ended budget-exhaustion task) and reports
   pass/fail per guardrail. This is the blueprint's stretch goal.
5. **Replay** tab — paste a past `run_id`, re-run its original task
   against the current prompt/model/guardrails, and see if anything
   regressed (status changed, cost or step count jumped).

## API

- `POST /run-agent` — `{task}` → runs the agent to completion or a halt condition.
- `GET /runs?limit=50` — recent runs.
- `GET /runs/{run_id}` — one run's metadata (status, cost, step count).
- `GET /runs/{run_id}/spans` — full trace for a run.
- `POST /replay/{run_id}` — re-run a past run's task, compare outcomes.
- `POST /redteam` — run the adversarial guardrail test suite.

## Notes on the illustrative pieces

- **Tracing** is SQLite-backed rather than a real OpenTelemetry/Langfuse
  export, to keep the project runnable with zero extra infra. See the
  integration-point comment at the bottom of `app/core/tracing.py` for
  exactly where to add an OTel span exporter or a Langfuse client call —
  the span shape (`name`, `input_data`, `output_data`, `metadata`,
  timestamps) maps directly onto either.
- **web_search** uses DuckDuckGo's free Instant Answer API (no key
  required) with an offline-safe fallback stub, and can be told (via
  `TOOLS_INJECT_TEST_CONTENT=1`) to append adversarial content to its
  results — this is what the red-team suite uses instead of needing a
  real compromised webpage.
- **calculator** uses a restricted AST-based evaluator (no `eval()`), so
  even if a prompt injection convinces the model to pass a malicious
  "expression," it's structurally incapable of executing arbitrary code —
  it just fails schema/safety validation and gets logged as a tool
  failure.
- Cost estimates in `COST_PER_1K_TOKENS` (`app/services/agent.py`) are
  illustrative — update to current published Grok pricing before treating
  the cost budget as a real dollar figure.
- This is written as a direct ReAct loop rather than through LangGraph,
  so every guardrail hook is visible in one file (`app/services/agent.py`)
  rather than hidden inside framework internals. Mapping it onto
  LangGraph: each iteration of the `while True` loop becomes a graph node
  (`reason_node`, `tool_node`), the guardrail checks become edge
  conditions, and `RunBudget` becomes part of the shared graph state.

## Project layout

```
app/
  core/
    tracing.py               # SQLite: runs + spans, OTel/Langfuse integration point noted
    budget.py                  # step/token/cost caps + circuit breaker (consecutive failures)
    pii_guardrail.py             # regex-based PII redaction
    injection_guardrail.py         # prompt-injection pattern scanning + untrusted-content wrapping
    schema_guardrail.py              # Pydantic validation of tool outputs
  services/
    tools.py                          # calculator (safe AST eval), web_search, final_answer
    agent.py                            # the ReAct loop, wraps every step in the guardrails above
    replay.py                             # re-run a past run's task, diff outcomes
    redteam.py                              # adversarial test suite (stretch goal)
  main.py                                    # FastAPI: run-agent, runs, replay, redteam
streamlit_app.py                              # UI: run + live trace, trace browser, red team, replay
```
