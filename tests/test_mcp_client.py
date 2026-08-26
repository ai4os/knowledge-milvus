"""
Test script demonstrating how a local MCP client bridges a cloud-hosted LLM
with a local MCP server over stdio.
"""

import asyncio
import json
import os
import sys

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI

load_dotenv()

# Cloud LLM configuration
LLM_MODEL = os.getenv("LLM_MODEL", "AI4EOSC/Qwen/Qwen3-14B")
LLM_BASE_URL = os.getenv("EMBEDDINGS_BASE_URL", "https://vllm.cloud.ai4eosc.eu/v1")
LITELLM_KEY = os.getenv("LITELLM_KEY")

openai_client = OpenAI(
    base_url=LLM_BASE_URL,
    api_key=LITELLM_KEY,
)


async def run_agent(user_prompt: str):
    # 1. Configure the local MCP server process (runs via stdio)
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["mcp_server.py"],
        env={**os.environ},
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
            print(f"🛠️  Discovered {len(mcp_tools.tools)} MCP tool(s):")
            for t in mcp_tools.tools:
                print(f"   - {t.name}: {t.description}")

            # 3. Format MCP tool definitions into OpenAI function-calling format for the cloud LLM
            openai_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": getattr(
                            tool, "input_schema", getattr(tool, "inputSchema", {})
                        ),
                    },
                }
                for tool in mcp_tools.tools
            ]

            # 4. Initiate conversation with the cloud-hosted LLM
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an assistant with access to tools for querying a knowledge database. "
                        "When needed, invoke the available tools to find relevant information."
                    ),
                },
                {"role": "user", "content": user_prompt},
            ]

            print(f"\n💬 User Prompt: {user_prompt}")

            # Run agent loop until LLM finishes (no more tool calls)
            MAX_ROUNDS = 5
            for round_idx in range(MAX_ROUNDS):
                print(f"☁️  Calling cloud LLM ({LLM_MODEL})...")
                response = openai_client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=messages,
                    tools=openai_tools,
                    tool_choice="auto",
                )

                assistant_message = response.choices[0].message
                messages.append(assistant_message)

                if not assistant_message.tool_calls:
                    print("\n" + "=" * 50)
                    print("🤖 Final Answer from LLM:")
                    print("=" * 50)
                    print(assistant_message.content)
                    break

                for tool_call in assistant_message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)

                    print(
                        f"\n⚙️  Cloud LLM requested tool execution: `{tool_name}` with args: {tool_args}"
                    )

                    result = await session.call_tool(tool_name, arguments=tool_args)

                    tool_output_parts = []
                    for content_item in result.content:
                        if hasattr(content_item, "text"):
                            tool_output_parts.append(content_item.text)
                        else:
                            tool_output_parts.append(str(content_item))
                    tool_output = "\n".join(tool_output_parts)

                    print(f"📥 Tool returned {len(tool_output)} characters of context.")

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_output,
                        }
                    )


if __name__ == "__main__":
    prompt = (
        "What collections do we have in Milvus? "
        "Then search for information on how AI4EOSC deploys pipelines in the relevant collection."
    )
    asyncio.run(run_agent(prompt))
