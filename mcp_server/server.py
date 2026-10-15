"""FinSight MCP Server — real MCP SDK implementation."""
from __future__ import annotations
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    'FinSight',
    description='Financial analysis tools powered by SEC EDGAR data, ML risk models, and RAG over 10-K filings',
)

# Import and register tools from tools module
from mcp_server.tools import register_tools
register_tools(mcp)

if __name__ == '__main__':
    mcp.run()  # stdio transport by default
