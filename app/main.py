from dotenv import load_dotenv
load_dotenv()  # loads .env for local `uvicorn` runs; no-op under Docker Compose

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.core.tracing import get_run, get_spans, init_db, recent_runs
from app.services.agent import run_agent
from app.services.redteam import run_red_team_suite
from app.services.replay import replay_run

app = FastAPI(
    title="Agentic System with Guardrails & Observability",
    description="A tool-using agent wrapped with tracing, PII/injection guardrails, "
                "budget enforcement, a circuit breaker, and a replay harness.",
    version="0.1.0",
)

init_db()


class RunAgentRequest(BaseModel):
    task: str


@app.post("/run-agent")
def run_agent_endpoint(req: RunAgentRequest):
    return run_agent(req.task)


@app.get("/runs")
def list_runs(limit: int = 50):
    return recent_runs(limit=limit)


@app.get("/runs/{run_id}")
def get_run_endpoint(run_id: str):
    run = get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@app.get("/runs/{run_id}/spans")
def get_spans_endpoint(run_id: str):
    return get_spans(run_id)


@app.post("/replay/{run_id}")
def replay_endpoint(run_id: str):
    try:
        return replay_run(run_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/redteam")
def redteam_endpoint():
    return run_red_team_suite()


@app.get("/health")
def health():
    return {"status": "ok"}
