"""MCP server (v2 part 3): lets an MCP client such as Claude Desktop ask questions of the Querycraft databases.

A thin client of the FastAPI service (serve/app.py), so the model, retriever and abstain model load once there.

    python -m querycraft.mcp_server          # stdio transport; QC_API_URL defaults to http://127.0.0.1:8000

Tools: ask(question, database, hint) and list_databases(). Queries are read only on the service side.
"""
import os

import httpx
from mcp.server import MCPServer

API = os.environ.get("QC_API_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT = float(os.environ.get("QC_API_TIMEOUT", "180"))

server = MCPServer("querycraft", instructions=(
    "Answers analytics questions over SQLite databases with a fine tuned text to SQL model. Call list_databases "
    "first, then ask. If the result says abstained, the model was not confident: show the SQL as a draft only."))


@server.tool()
def list_databases() -> list[dict]:
    """Databases that can be queried, with their tables."""
    r = httpx.get(f"{API}/databases", timeout=30)
    r.raise_for_status()
    return r.json()


@server.tool()
def ask(question: str, database: str, hint: str = "") -> dict:
    """Answer a question about one database. Returns the SQL, up to 50 result rows, the tables used, a
    calibrated confidence and whether the model abstained (not confident enough to answer)."""
    r = httpx.post(f"{API}/ask", json={"question": question, "database": database, "hint": hint}, timeout=TIMEOUT)
    if r.status_code >= 400:
        return {"error": r.json().get("detail", r.text)}
    d = r.json()
    return {k: d[k] for k in ("sql", "columns", "confidence", "abstained", "tables", "error")} | {"rows": d["rows"][:50]}


def main():
    server.run("stdio")


if __name__ == "__main__":
    main()
