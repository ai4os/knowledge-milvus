# AI4EOSC Knowledge Milvus

This repo contains the [workflow](./main.py) to save knowledge into the Milvus Vector Database:

* load the content retrieved by the [Ai4EOSC Knowledge extractors](https://github.com/ai4os/knowledge-extractors),
* delete from Milvus DB the old content that no longer exists in the web,
* for each webpage, hash the content and check against previous hash,
* if the content was updated, chunk it, create embeddings and save in Milvus DB,
* save new hashes

#### Usage

* Install the requirements:

```
pip install -r requirements.txt
```

* Define you environment variables:

```
LITELLM_KEY=**********************************
MILVUS_URI=https://milvus.k8s.cloud.ai4eosc.eu
MILVUS_PWD=***********************************
```

* Generate and save the embeddings:

```
python main.py
```

To test the Milvus is correctly up and running.

```
python tests/health.py
```

#### Implementation notes

* While initially we considered integrating the Milvus vector store directly in LiteLLM (`utils/litellm_milvus_create.py`), LiteLLM seemed to be unable to use it to perform searches (`tests/test_litellm_search*.py`). So we decided to put it behind an MCP server and integrate the MCP server instead in LiteLLM.


#### MCP Server

You can run the Model Context Protocol (MCP) server to allow LLMs and MCP clients to query the Milvus vector database:

```bash
python mcp_server.py
```

Available tools:
* `list_collections()`: Lists all collections available in the Milvus database.
* `search(query, collections, limit=5)`: Performs vector search across one or more collections and returns the top matching document chunks with their similarity scores and metadata (`text`, `filename`, `url`).

To test an end-to-end tool-calling loop using a cloud-hosted LLM and the local MCP server:

```bash
python tests/test_mcp_client.py
```
