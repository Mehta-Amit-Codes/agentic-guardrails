"""
Tool-output schema validation. A tool that returns malformed data (wrong
types, missing fields, an error masquerading as a result) shouldn't be
silently handed to the LLM as if it were a clean observation -- that's
how agents end up reasoning over garbage. Every tool's raw output is
validated against its declared schema before it's added to the agent's
context.
"""
from typing import Any

from pydantic import BaseModel, ValidationError


class ToolOutputError(Exception):
    """Raised when a tool's raw output fails schema validation."""
    def __init__(self, tool_name: str, errors: str):
        self.tool_name = tool_name
        self.errors = errors
        super().__init__(f"Tool '{tool_name}' output failed validation: {errors}")


class CalculatorResult(BaseModel):
    expression: str
    result: float


class WebSearchResult(BaseModel):
    query: str
    results: list[dict]  # each dict expected to have "title" and "snippet"


class FinalAnswer(BaseModel):
    answer: str
    confidence: float  # 0.0-1.0, self-reported


TOOL_SCHEMAS: dict[str, type[BaseModel]] = {
    "calculator": CalculatorResult,
    "web_search": WebSearchResult,
    "final_answer": FinalAnswer,
}


def validate_tool_output(tool_name: str, raw_output: dict) -> dict:
    """Raises ToolOutputError on failure; returns the validated dict on success."""
    schema = TOOL_SCHEMAS.get(tool_name)
    if schema is None:
        return raw_output  # no schema registered for this tool -- pass through
    try:
        validated = schema(**raw_output)
        return validated.model_dump()
    except ValidationError as e:
        raise ToolOutputError(tool_name, str(e)) from e


def describe_schema(tool_name: str) -> dict[str, Any] | None:
    schema = TOOL_SCHEMAS.get(tool_name)
    return schema.model_json_schema() if schema else None
