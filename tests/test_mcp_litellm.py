"""
Use a registered LiteLLM MCP to list collections.
"""

import json
import os

from dotenv import load_dotenv
import requests

load_dotenv()

BASE_URL = "https://vllm.cloud.ai4eosc.eu"
MCP_SERVER_NAME = "knowledge_milvus_private"
LITELLM_KEY = os.environ["LITELLM_KEY"]

headers = {
    "Authorization": f"Bearer {LITELLM_KEY}",
    "Content-Type": "application/json",
}

# 1. List tools
list_tools_url = f"{BASE_URL}/mcp-rest/tools/list"
params = {"mcp_server_name": MCP_SERVER_NAME}

print(f"Fetching tools for MCP server: {MCP_SERVER_NAME}...")
response = requests.get(list_tools_url, headers=headers, params=params)
print(f"List tools status code: {response.status_code}")
tools_data = response.json()
print(json.dumps(tools_data, indent=2))

# 2. Call list_collections passing also a fake x-user-token
call_tool_url = f"{BASE_URL}/mcp-rest/tools/call"
call_headers = {
    **headers,
    "x-user-token": "fake-jwt-token",
}
call_payload = {
    "server_id": MCP_SERVER_NAME,
    "name": "list_collections",
    "arguments": {},
}

print(f"\nCalling list_collections on {MCP_SERVER_NAME}...")
call_response = requests.post(
    call_tool_url,
    headers=call_headers,
    json=call_payload,
)

print(f"Call tool status code: {call_response.status_code}")
call_data = call_response.json()
print(json.dumps(call_data, indent=2))
