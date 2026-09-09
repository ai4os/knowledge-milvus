"""
Test private collections with local MCP server over stdio and user OIDC token.
"""

import asyncio
import os
import subprocess
import sys

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

# Retrieve token with oidc-token command
result = subprocess.run(
    ["oidc-token", "ai4os-keycloak"], capture_output=True, text=True, check=True
)
token = result.stdout.strip()

if not token:
    raise PermissionError(
        "PAPI needs to retrieve an OIDC token either using oidc-agent or ENV variables."
    )


async def main():
    # 1. Configure the local MCP server process (runs via stdio)
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["mcp_server.py"],
        env={**os.environ, "X_USER_TOKEN": token},
    )

    print(
        f"🚀 Connecting to local MCP server via stdio ({sys.executable} mcp_server.py)..."
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize MCP session
            await session.initialize()

            # 2. Retrieve available tools from the local MCP server
            mcp_tools = await session.list_tools()
            print(f"\n🛠️  Discovered {len(mcp_tools.tools)} MCP tool(s):")
            for t in mcp_tools.tools:
                print(f"   - {t.name}: {t.description}")

            # 3. Call list_collections
            print("\n📂 Calling list_collections for private collections...")
            res = await session.call_tool("list_collections", arguments={"type": "private"})
            for content in res.content:
                if hasattr(content, "text"):
                    print(content.text)
                else:
                    print(content)


if __name__ == "__main__":
    asyncio.run(main())