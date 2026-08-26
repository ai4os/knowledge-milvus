import json
import os

from dotenv import load_dotenv
import requests

load_dotenv()

vector_store_id = "ai4eosc_docs"
query = "How do I register in AI4EOSC?"
filters = None  # e.g. {"type": "eq", "key": "category", "value": "support"}
max_num_results = 10

base_url = "https://vllm.cloud.ai4eosc.eu/v1"
api_key = os.getenv("LITELLM_KEY")

url = f"{base_url.rstrip('/')}/vector_stores/{vector_store_id}/search"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
}
payload = {
    "query": query,
    "max_num_results": max_num_results,
}
if filters:
    payload["filters"] = filters

response = requests.post(url, headers=headers, json=payload)
response.raise_for_status()

results = response.json()
print(json.dumps(results, indent=2))
