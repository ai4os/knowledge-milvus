"""
Create the vector store in LiteLLM.
"""

import json
import os
import requests

from dotenv import load_dotenv


load_dotenv()

VECTOR_STORE_ID = "ai4eosc_docs"
MILVUS_URI = os.environ["MILVUS_URI"]
MILVUS_PWD = os.environ["MILVUS_PWD"]
LITELLM_KEY = os.environ["LITELLM_KEY"]
BASE_URL = "https://vllm.cloud.ai4eosc.eu"
EMBEDDING_MODEL = "AI4EOSC/Qwen/Qwen3-Embedding-4B"
EMBEDDING_MODEL = f"openai/{EMBEDDING_MODEL}"

milvus_api_key = f"root:{MILVUS_PWD}"
embedding_config = {
    "api_base": f"{BASE_URL}/v1",
    "api_key": LITELLM_KEY,
}

litellm_params = {
    "api_base": MILVUS_URI,
    "api_key": milvus_api_key,
    "embedding_model": EMBEDDING_MODEL,
    "litellm_embedding_model": EMBEDDING_MODEL,
    "embedding_config": embedding_config,
    "litellm_embedding_config": embedding_config,
    "milvus_text_field": "text",
    "milvus_vector_field": "vector",
}

payload = {
    "vector_store_id": VECTOR_STORE_ID,
    "custom_llm_provider": "milvus",
    "vector_store_name": VECTOR_STORE_ID,
    "litellm_params": litellm_params,
}

headers = {
    "Authorization": f"Bearer {LITELLM_KEY}",
    "Content-Type": "application/json",
}

response = requests.post(
    f"{BASE_URL}/vector_store/new",
    headers=headers,
    json=payload,
)

print(f"Status code: {response.status_code}")
try:
    print(json.dumps(response.json(), indent=2))
except Exception:
    print(response.text)
