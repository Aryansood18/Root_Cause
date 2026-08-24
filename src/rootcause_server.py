"""
RootCause MCP server — a thin wrapper around core.py.

Exposes one MCP tool, `analyze_bug`, over stdio. All retrieval/rerank/generation logic
lives in core.py so the dashboard and the server share exactly one implementation. This
file only adapts that pipeline to the MCP contract: take a query string, return a JSON
string (the model's answer, or a JSON error object on failure).

Run:
    python src/rootcause_server.py

It speaks stdio, so it's meant to be launched by an MCP client, not visited in a browser.
See the README for the client registration snippet.
"""

import json

from mcp.server.fastmcp import FastMCP

import core

mcp = FastMCP("RootCause")


@mcp.tool()
def analyze_bug(query: str) -> str:
    """
    Analyze a bug using RAG + reranking + confidence-gated fallback.
    Returns structured JSON: {root_cause, fix, confidence, examples_used}.
    """
    try:
        return core.analyze_bug(query)["answer"]
    except Exception as e:
        return json.dumps({"error": str(e)})


if __name__ == "__main__":
    mcp.run(transport="stdio")
