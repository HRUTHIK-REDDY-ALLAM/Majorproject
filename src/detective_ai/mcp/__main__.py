"""Run the Detective AI MCP server over stdio.

Usage:
    python -m detective_ai.mcp
"""

from detective_ai.mcp.tools import mcp_server
from detective_ai.storage.database import db

if __name__ == "__main__":
    db.init_db()
    mcp_server.run()
