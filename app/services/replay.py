"""
Replay harness. Stored traces (core/tracing.py) let you take a past run's
original task and re-run it after changing the system prompt, swapping
models, or updating a guardrail -- then diff the outcome against what
actually happened, to catch regressions before they reach production.

This replays the *task*, not a scripted tool-call sequence: the agent
makes its own decisions again, since the point is to see whether the new
prompt/model still behaves acceptably, not to force the same tool calls.
"""
from app.core.tracing import get_run, get_spans
from app.services.agent import run_agent


def replay_run(original_run_id: str) -> dict:
    original = get_run(original_run_id)
    if original is None:
        raise ValueError(f"No run found with id {original_run_id}")

    original_spans = get_spans(original_run_id)
    new_result = run_agent(original["task"])

    return {
        "original_run_id": original_run_id,
        "new_run_id": new_result["run_id"],
        "original": {
            "status": original["status"],
            "step_count": original["step_count"],
            "total_cost_usd": original["total_cost_usd"],
            "final_answer": _final_answer_from_spans(original_spans),
        },
        "new": {
            "status": new_result["status"],
            "step_count": new_result["budget"]["steps_used"],
            "total_cost_usd": new_result["budget"]["cost_used_usd"],
            "final_answer": new_result["answer"],
        },
        "regression_flags": _compare(original, original_spans, new_result),
    }


def _final_answer_from_spans(spans: list[dict]) -> str | None:
    for span in spans:
        if span["span_type"] == "final_answer":
            import json
            return json.loads(span["output_data"]).get("answer")
    return None


def _compare(original: dict, original_spans: list[dict], new_result: dict) -> list[str]:
    flags = []
    if original["status"] == "completed" and new_result["status"] != "completed":
        flags.append(f"Regression: original run completed, replay ended with status '{new_result['status']}'")
    if new_result["budget"]["steps_used"] > original["step_count"] * 1.5:
        flags.append(
            f"Step count increased significantly: {original['step_count']} -> "
            f"{new_result['budget']['steps_used']}"
        )
    if new_result["budget"]["cost_used_usd"] > (original["total_cost_usd"] or 0) * 1.5:
        flags.append(
            f"Cost increased significantly: ${original['total_cost_usd']:.4f} -> "
            f"${new_result['budget']['cost_used_usd']:.4f}"
        )
    if not flags:
        flags.append("No regressions detected.")
    return flags
