"""
PII redaction. Applied to tool outputs and any external content before it
reaches the LLM -- an agent that reads a webpage or a file shouldn't leak
whatever emails, phone numbers, SSNs, or card numbers happen to appear in
that content back into the model's context (and from there, potentially
into logs, traces, or the final answer).

Regex-based and intentionally conservative (prefers over-redacting to
under-redacting). For production, pair with a proper PII detection
library (Presidio, etc.) -- this covers the common, high-confidence
patterns without extra dependencies.
"""
import re

PATTERNS = {
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "PHONE": re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    "IP_ADDRESS": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}


def redact_pii(text: str) -> tuple[str, list[str]]:
    """Returns (redacted_text, list_of_pii_types_found)."""
    found = []
    redacted = text
    for label, pattern in PATTERNS.items():
        if pattern.search(redacted):
            found.append(label)
            redacted = pattern.sub(f"[REDACTED_{label}]", redacted)
    return redacted, found
