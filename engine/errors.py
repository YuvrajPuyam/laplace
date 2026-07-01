"""Structured tool errors per schemas/tools.md: never prose-only.

Every error crossing a transport boundary serializes to
{"error": {"code": ..., "message": ..., "details": {...}}}.
"""

from __future__ import annotations


class ToolError(Exception):
    def __init__(self, code: str, message: str, details: dict | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}

    def envelope(self) -> dict:
        return {"error": {"code": self.code, "message": self.message,
                          "details": self.details}}
