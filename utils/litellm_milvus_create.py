"""
Create the vector store in LiteLLM.
"""

import json
import os
import requests

from dotenv import load_dotenv


load_dotenv()

vector_store_id = "ai4eosc_docs"
milvus_uri = os.getenv("MILVUS_URI")
milvus_pwd = os.getenv("MILVUS_PWD")
litellm_key = os.getenv("LITELLM_KEY")
base_url = "https://vllm.cloud.ai4eosc.eu"
embedding_model = "AI4EOSC/Qwen/Qwen3-Embedding-4B"
embedding_model = f"openai/{embedding_model}"

milvus_api_key = f"root:{milvus_pwd}"
embedding_config = {
    "api_base": f"{base_url}/v1",
    "api_key": litellm_key,
}

litellm_params = {
    "api_base": milvus_uri,
    "api_key": milvus_api_key,
    "embedding_model": embedding_model,
    "litellm_embedding_model": embedding_model,
    "embedding_config": embedding_config,
    "litellm_embedding_config": embedding_config,
    "milvus_text_field": "text",
    "milvus_vector_field": "vector",
}

payload = {
    "vector_store_id": vector_store_id,
    "custom_llm_provider": "milvus",
    "vector_store_name": vector_store_id,
    "litellm_params": litellm_params,
}

headers = {
    "Authorization": f"Bearer {litellm_key}",
    "Content-Type": "application/json",
}

response = requests.post(
    f"{base_url}/vector_store/new",
    headers=headers,
    json=payload,
)

print(f"Status code: {response.status_code}")
try:
    print(json.dumps(response.json(), indent=2))
except Exception:
    print(response.text)
