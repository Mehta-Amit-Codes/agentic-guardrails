"""
Streamlit UI for the Agentic System with Guardrails & Observability.

Tabs:
  - Run Agent: give it a task, watch the trace unfold step by step
    (thought -> tool call -> guardrail checks -> observation)
  - Trace Browser: inspect any past run's full span history
  - Red Team: run the adversarial test suite, see pass/fail per guardrail
  - Replay: re-run a past run's task against the current prompt/model,
    compare outcomes

Run:
    streamlit run streamlit_app.py
"""
import json
import os

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.environ.get("AGENT_API_URL", "http://localhost:8003")

st.set_page_config(page_title="Agentic Guardrails", page_icon="🛡️", layout="wide")
st.title("🛡️ Agentic System with Guardrails & Observability")
st.caption(f"API: {API_BASE_URL} · Tracing + PII/injection guardrails + budget enforcement + replay")

tab_run, tab_trace, tab_redteam, tab_replay = st.tabs(
    ["🏃 Run Agent", "🔍 Trace Browser", "🎯 Red Team", "🔁 Replay"]
)

SPAN_ICONS = {
    "thought": "💭", "tool_call": "🔧", "final_answer": "✅", "guardrail_block": "🚫",
}


def render_spans(spans: list[dict]) -> None:
    for span in spans:
        icon = SPAN_ICONS.get(span["span_type"], "•")
        metadata = json.loads(span["metadata"]) if span["metadata"] else {}

        with st.container(border=True):
            st.markdown(f"{icon} **Step {span['step_index']}** — `{span['span_type']}` / `{span['name']}`")

            if span["span_type"] == "thought":
                output = json.loads(span["output_data"])
                if output.get("content"):
                    st.write(output["content"])
                if output.get("tool_calls"):
                    for tc in output["tool_calls"]:
                        st.code(f"{tc['name']}({tc['arguments']})", language="json")
                st.caption(
                    f"tokens: {metadata.get('tokens_in', 0)} in / {metadata.get('tokens_out', 0)} out "
                    f"· cost: ${metadata.get('cost_usd', 0):.6f}"
                )

            elif span["span_type"] == "tool_call":
                col1, col2 = st.columns(2)
                with col1:
                    st.caption("Input")
                    st.code(json.dumps(json.loads(span["input_data"]), indent=2), language="json")
                with col2:
                    st.caption("Validated output")
                    st.code(json.dumps(json.loads(span["output_data"]), indent=2), language="json")

                flags = []
                if metadata.get("failed"):
                    flags.append("❌ tool failed")
                if metadata.get("pii_types_found"):
                    flags.append(f"🔒 PII redacted: {', '.join(metadata['pii_types_found'])}")
                if metadata.get("injection_hits_in_output") or metadata.get("injection_hits_in_args"):
                    flags.append("⚠️ prompt-injection pattern detected in tool content")
                if flags:
                    st.warning(" · ".join(flags))
                else:
                    st.caption("✓ no guardrail flags")

            elif span["span_type"] == "final_answer":
                output = json.loads(span["output_data"])
                st.success(f"**Answer:** {output['answer']}")
                st.caption(f"Confidence: {output['confidence']:.2f}")


# ---- Run Agent -----------------------------------------------------------
with tab_run:
    st.subheader("Give the agent a task")
    default_task = "What is 47 * 12, and can you also search for 'FastAPI' and summarize the top result?"
    task = st.text_area("Task", value=default_task, height=80)

    col1, col2 = st.columns(2)
    with col1:
        st.caption(f"Max steps / tokens / cost: env-configured "
                    f"(defaults: {os.environ.get('AGENT_MAX_STEPS', 10)} steps, "
                    f"{os.environ.get('AGENT_MAX_TOKENS', 20000)} tokens, "
                    f"${os.environ.get('AGENT_MAX_COST_USD', 0.50)})")

    if st.button("Run agent", type="primary") and task:
        with st.spinner("Agent is working — watch the trace below as it completes..."):
            resp = requests.post(f"{API_BASE_URL}/run-agent", json={"task": task}, timeout=120)

        if resp.ok:
            result = resp.json()
            status_colors = {"completed": "success", "budget_exceeded": "warning", "halted": "error"}
            getattr(st, status_colors.get(result["status"], "info"))(
                f"Run **{result['status']}** — run_id: `{result['run_id']}`"
            )
            if result.get("halt_reason"):
                st.error(f"Halt reason: {result['halt_reason']}")
            if result.get("answer"):
                st.markdown(f"### Answer\n{result['answer']}")

            budget = result["budget"]
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Steps", f"{budget['steps_used']}/{budget['max_steps']}")
            col2.metric("Tokens", f"{budget['tokens_used']}/{budget['max_tokens']}")
            col3.metric("Cost", f"${budget['cost_used_usd']:.4f}/${budget['max_cost_usd']}")
            col4.metric("Consecutive failures", budget["consecutive_failures"])

            st.divider()
            st.subheader("Trace")
            spans_resp = requests.get(f"{API_BASE_URL}/runs/{result['run_id']}/spans")
            if spans_resp.ok:
                render_spans(spans_resp.json())
        else:
            st.error(f"Failed: {resp.status_code} {resp.text}")

# ---- Trace Browser ------------------------------------------------------
with tab_trace:
    st.subheader("Past runs")
    resp = requests.get(f"{API_BASE_URL}/runs", params={"limit": 30})
    if resp.ok:
        runs = resp.json()
        if not runs:
            st.info("No runs yet — try the Run Agent tab.")
        else:
            df = pd.DataFrame(runs)[["run_id", "task", "status", "step_count", "total_cost_usd", "started_at"]]
            st.dataframe(df, use_container_width=True)

            selected_run_id = st.selectbox("Inspect a run", [r["run_id"] for r in runs])
            if selected_run_id and st.button("Load trace"):
                spans_resp = requests.get(f"{API_BASE_URL}/runs/{selected_run_id}/spans")
                if spans_resp.ok:
                    render_spans(spans_resp.json())

# ---- Red Team -------------------------------------------------------
with tab_redteam:
    st.subheader("Adversarial guardrail test suite")
    st.caption(
        "Runs the agent against tasks designed to trigger each guardrail "
        "(prompt injection in tool output, PII in tool output, an unsafe calculator "
        "expression, and an open-ended task that should hit the budget cap) and checks "
        "whether the guardrail actually caught it."
    )
    if st.button("Run red team suite", type="primary"):
        with st.spinner("Running adversarial test cases (this makes several agent calls)..."):
            resp = requests.post(f"{API_BASE_URL}/redteam", timeout=300)
        if resp.ok:
            results = resp.json()
            passed = sum(1 for r in results if r["passed"])
            st.metric("Guardrails passing", f"{passed}/{len(results)}")

            for r in results:
                icon = "✅" if r["passed"] else "❌"
                with st.container(border=True):
                    st.markdown(f"{icon} **{r['name']}** — guardrail: `{r['guardrail']}`")
                    st.caption(f"Task: {r['task']}")
                    st.caption(f"Expected: {r['expected']}")
                    st.caption(f"Run status: {r['run_status']} · run_id: `{r['run_id']}`")
        else:
            st.error(f"Failed: {resp.status_code} {resp.text}")

# ---- Replay -----------------------------------------------------------
with tab_replay:
    st.subheader("Replay a past run")
    st.caption(
        "Re-runs a past run's original task against the CURRENT prompt/model/guardrails, "
        "and flags regressions (status changed, step count or cost jumped significantly)."
    )
    run_id_to_replay = st.text_input("Run ID to replay")
    if st.button("Replay", type="primary") and run_id_to_replay:
        with st.spinner("Re-running the task..."):
            resp = requests.post(f"{API_BASE_URL}/replay/{run_id_to_replay}")
        if resp.ok:
            data = resp.json()
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Original")
                st.json(data["original"])
            with col2:
                st.markdown("### Replay")
                st.json(data["new"])

            st.divider()
            for flag in data["regression_flags"]:
                if flag.startswith("Regression") or "increased" in flag:
                    st.warning(flag)
                else:
                    st.success(flag)
        else:
            st.error(f"Failed: {resp.status_code} {resp.text}")
