"""Reject upstream lexical tokens at the raw-to-CHSL boundary."""
from __future__ import annotations
import json
import re
from pathlib import Path

TOKENS = json.loads((Path(__file__).parent / "quarantine_map.json").read_text())[
    "prohibited_export_tokens"
]
PATTERNS = tuple(
    re.compile(r"(?<![a-z0-9])" + re.escape(token) + r"(?![a-z0-9])", re.I)
    for token in TOKENS
)

def assert_clean(text: str) -> None:
    if any(pattern.search(text) for pattern in PATTERNS):
        raise ValueError("CHSL-v1 outbound lexical quarantine violation")
