"""
MCP server to interact with the Milvus database
"""

import argparse
import os
from typing import Any

from dotenv import load_dotenv
from mcp.server import MCPServer
from openai import OpenAI
from pymilvus import MilvusClient

load_dotenv()

# Configuration
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "AI4EOSC/Qwen/Qwen3-Embedding-4B")
EMBEDDINGS_BASE_URL = os.getenv(
    "EMBEDDINGS_BASE_URL", "https://vllm.cloud.ai4eosc.eu/v1"
)

# Initialize Clients
openai_client = OpenAI(
    base_url=EMBEDDINGS_BASE_URL,
    api_key=os.getenv("LITELLM_KEY"),
)

milvus_uri = os.getenv("MILVUS_URI")
milvus_pwd = os.getenv("MILVUS_PWD")
milvus_client = MilvusClient(
    uri=milvus_uri,
    user="root",
    password=milvus_pwd,
)

# Initialize MCP Server
mcp = MCPServer("knowledge-milvus")


def get_embedding(text: str) -> list[float]:
    """Generate vector embedding for the input text."""
    response = openai_client.embeddings.create(
        input=text,
        model=EMBEDDINGS_MODEL,
    )
    return response.data[0].embedding


@mcp.tool()
def list_collections() -> list[str]:
    """
    Retrieve all available collections from the Milvus database.

    Returns:
        List of collection names.
    """
    return milvus_client.list_collections()


@mcp.tool()
def search(
    query: str,
    collections: list[str] | str,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """
    Perform vector RAG searches across one or more Milvus collections.

    Args:
        query: Search query text to find relevant document chunks.
        collections: One valid collection name (str) or a list of collection names (list[str]) from list_collections.
        limit: Maximum number of top results to return (default: 5).

    Returns:
        List of retrieved result objects containing text chunk, score, collection name,
        and metadata (e.g. filename, url).
    """
    if isinstance(collections, str):
        collection_list = [collections]
    else:
        collection_list = list(collections)

    if not collection_list:
        return []

    # Validate collections against existing collections to avoid hard crashes
    available_collections = set(milvus_client.list_collections())
    invalid_collections = [c for c in collection_list if c not in available_collections]
    if invalid_collections:
        return [
            {
                "error": f"Collection(s) not found: {invalid_collections}. Available collections are: {list(available_collections)}"
            }
        ]

    embedding = get_embedding(query)
    all_results: list[dict[str, Any]] = []

    for col in collection_list:
        try:
            search_res = milvus_client.search(
                collection_name=col,
                data=[embedding],
                limit=limit,
                output_fields=["text", "filename", "url"],
            )
        except Exception as e:
            raise RuntimeError(f"Error searching collection '{col}': {e}") from e

        if search_res and len(search_res) > 0:
            for hit in search_res[0]:
                entity = (
                    hit.get("entity", {})
                    if isinstance(hit, dict)
                    else getattr(hit, "entity", {})
                )
                score = (
                    hit.get("distance", hit.get("score", 0.0))
                    if isinstance(hit, dict)
                    else getattr(hit, "distance", getattr(hit, "score", 0.0))
                )
                doc_id = (
                    hit.get("id") if isinstance(hit, dict) else getattr(hit, "id", None)
                )

                all_results.append(
                    {
                        "id": doc_id,
                        "collection": col,
                        "score": score,
                        "text": entity.get("text", "")
                        if isinstance(entity, dict)
                        else getattr(entity, "text", ""),
                        "filename": entity.get("filename", "")
                        if isinstance(entity, dict)
                        else getattr(entity, "filename", ""),
                        "url": entity.get("url", "")
                        if isinstance(entity, dict)
                        else getattr(entity, "url", ""),
                    }
                )

    # Sort results by similarity score descending
    all_results.sort(key=lambda x: x["score"], reverse=True)

    return all_results[:limit]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Milvus MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="Transport protocol to use (default: stdio)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind for HTTP/SSE transport (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind for HTTP/SSE transport (default: 8000)",
    )
    args = parser.parse_args()

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport=args.transport, host=args.host, port=args.port)
