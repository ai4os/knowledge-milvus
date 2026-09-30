import os

from dotenv import load_dotenv
from pymilvus import MilvusClient
from openai import OpenAI

load_dotenv()

collection_name = "openaire_docs"
# collection_name = "ai4eosc_docs"


# Init embeddings model
openai_client = OpenAI(
    base_url="https://vllm.cloud.ai4eosc.eu/v1",
    api_key=os.getenv("LITELLM_KEY"),
)

# Init Milvus
# milvus_client = MilvusClient(uri="./milvus.db")
milvus_uri = os.getenv("MILVUS_URI")
milvus_pwd = os.getenv("MILVUS_PWD")
milvus_client = MilvusClient(uri=milvus_uri, user="root", password=milvus_pwd)


def get_rag_response(user_question):
    # STEP 1: Embed the user question
    # Must use the same model used to index your Milvus data
    embedding = (
        openai_client.embeddings.create(
            input=user_question, model="AI4EOSC/Qwen/Qwen3-Embedding-4B"
        )
        .data[0]
        .embedding
    )

    # STEP 2: Search Milvus
    search_res = milvus_client.search(
        collection_name=collection_name,
        data=[embedding],
        limit=3,  # Number of chunks to retrieve
        output_fields=["text"],  # Ensure your text field is returned
    )

    # STEP 3: Combine retrieved text into a single context string
    retrieved_text = [res["entity"]["text"] for res in search_res[0]]
    context = "\n\n".join(retrieved_text)

    # STEP 4: Feed to OpenAI Chat Completions
    response = openai_client.chat.completions.create(
        model="AI4EOSC/mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Use the provided context to answer the question.",
            },
            {
                "role": "user",
                "content": f" #### Context:\n{context}\n\n#### Question: {user_question}",
            },
        ],
    )

    return response.choices[0].message.content


# print(get_rag_response("What is John's favourite fruit?"))
print(get_rag_response("How does AI4EOSC deploys pipelines?"))
