import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

vector_store_id = "ai4eosc_docs"
query = "How do I register in AI4EOSC?"
filters = None  # e.g. {"type": "eq", "key": "category", "value": "support"}
max_num_results = 10

client = OpenAI(
    base_url="https://vllm.cloud.ai4eosc.eu/v1",
    api_key=os.getenv("LITELLM_KEY"),
)

payload = {
    "query": query,
    "max_num_results": max_num_results,
}
if filters:
    payload["filters"] = filters

results = client.post(
    f"/vector_stores/{vector_store_id}/search",
    body=payload,
    cast_to=dict,
)

print(json.dumps(results, indent=2))
