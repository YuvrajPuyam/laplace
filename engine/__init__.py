"""WS2 — engine: tool handlers, budget ledger, episode lifecycle, HTTP/MCP.

The eight tools in schemas/tools.md are implemented as pure-ish handlers over
an Episode object (engine/tools.py). The FastAPI service (engine/api.py) and
the MCP server (engine/mcp_server.py) are thin transports over the same
handlers — one implementation, two protocols.
"""
