"""
Upload to Milvus the docs from the scrappers.
"""

import hashlib
import json
import os
from pathlib import Path
import re
import time

from dotenv import load_dotenv
import frontmatter
from openai import OpenAI
from pymilvus import MilvusClient, DataType
import tqdm

import utils.text as text_utils

load_dotenv()

EMBEDDINGS_SIZE = 2560  # specific to Qwen3-Embeddings-4B
CHUNKSIZE, OVERLAP = 2048, 256  # good compromise between context and detail
# CHUNKSIZE, OVERLAP = 1024, 128  # better for fact-checking (i.e. technical docs)
BATCH_SIZE = 10  # batch size for computing embeddings in OpenAI API calls
EMBEDDINGS_MODEL = "AI4EOSC/Qwen/Qwen3-Embedding-4B"

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


SCRAPERS_DOWNLOADS = Path(
    "/home/iheredia/ignacio/projects/arena/repos/knowledge-extractor/downloads"
)
for collection_pth in SCRAPERS_DOWNLOADS.iterdir():
    if not collection_pth.is_dir():
        continue

    start_time = time.time()

    md_files = list(collection_pth.glob("*.md"))
    collection_name = re.sub(r"[^a-zA-Z0-9_]", "_", collection_pth.name)
    # collection name can only contain numbers, letters and underscores
    print(f"📚 Processing collection: {collection_name}")

    if collection_name not in ["imagine", "papi", "github"]:
        continue

    # Load the hashes computed from a previous run
    hash_pth = f"./hashes/{collection_name}.json"
    os.makedirs(os.path.dirname(hash_pth), exist_ok=True)
    if Path(hash_pth).exists():
        with open(hash_pth) as f:
            hashes = json.load(f)
    else:
        hashes = {}

    if collection_name not in milvus_client.list_collections():
        # Create the collection schema
        schema = milvus_client.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        # Embeddings vector
        schema.add_field(
            field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=EMBEDDINGS_SIZE
        )
        # Original text
        schema.add_field(
            field_name="text",
            datatype=DataType.VARCHAR,
            max_length=CHUNKSIZE * 4,
            # Measured in bytes but chunksize is in characters so we include a safe
            # buffer for UTF-8 characters (which take 4 bytes)
        )
        # Filename
        schema.add_field(
            field_name="filename", datatype=DataType.VARCHAR, max_length=512
        )
        # URL
        schema.add_field(field_name="url", datatype=DataType.VARCHAR, max_length=2048)

        # Create an index for fast search
        index_params = milvus_client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            metric_type="COSINE",
            index_type="HNSW",  # this is the gold standard for speed/accuracy trade-offs
            # index_type="AUTOINDEX",
        )

        milvus_client.create_collection(
            collection_name=collection_name, schema=schema, index_params=index_params
        )

    # Ensure the collection is loaded into memory before querying
    # (collections can be unloaded after server restarts or memory pressure)
    milvus_client.load_collection(collection_name=collection_name)

    # Iterate over Milvus to remove old content that no longer exists
    # Get all filenames from Milvus collection
    iterator = milvus_client.query_iterator(
        collection_name=collection_name,
        filter="",  # Get everything
        output_fields=["filename"],
        batch_size=1000,  # Process 1000 records at a time
    )

    print("🔍 Scanning Milvus for existing files...")
    milvus_files = set()
    while True:
        # Get the next batch of results
        result_batch = iterator.next()
        if not result_batch:
            break

        for entry in result_batch:
            milvus_files.add(entry["filename"])

    # Now safely calculate what to delete
    current_files = {f.name for f in md_files}
    files_to_delete = milvus_files - current_files
    for f in files_to_delete:
        print(f"🔴 [Milvus] Deleting obsolete file: {f}")
        milvus_client.delete(
            collection_name=collection_name, filter=f'filename == "{f}"'
        )
        if f in hashes:
            del hashes[f]

    iterator.close()

    # Add new content to Milvus
    for file in md_files:
        fname = file.name

        with open(file, "r") as f:
            # Separate the body from the YAML frontmatter
            post = frontmatter.load(f)
            metadata = post.metadata
            body = post.content

        file_hash = hashlib.md5(body.encode()).hexdigest()

        if "source_url" not in metadata:
            raise ValueError(f"{fname}: missing 'source_url' in frontmatter")

        # Check if file is unchanged
        if fname in hashes and hashes[fname] == file_hash:
            continue

        # If the file exists in old_hashes but hash had changed, we must clear previous vectors
        if fname in hashes:
            print(f"🔵 [Milvus] Updating file: {fname}")
            milvus_client.delete(
                collection_name=collection_name, filter=f'filename == "{fname}"'
            )
        else:
            print(f"🟢 [Milvus] Adding new file: {fname}")

        # Compute chunks
        chunks = text_utils.semantic_chunking(
            body, chunk_size=CHUNKSIZE, overlap=OVERLAP
        )

        # Process chunks in batches to reduce API overhead
        data_to_insert = []
        for i in tqdm.tqdm(
            range(0, len(chunks), BATCH_SIZE), desc="Computing embeddings"
        ):
            batch_chunks = chunks[i : i + BATCH_SIZE]

            try:
                # Batch API Call
                response = openai_client.embeddings.create(
                    model=EMBEDDINGS_MODEL, input=batch_chunks
                )

                # Match embeddings to their text chunks
                for j, embedding_data in enumerate(response.data):
                    data_to_insert.append(
                        {
                            "vector": embedding_data.embedding,
                            "text": batch_chunks[j],
                            "filename": fname,
                            "url": metadata["source_url"],
                        }
                    )

            except Exception as e:
                print(f"Error computing embeddings for {fname}: {e}")

                # If at least one of the calls raises error, save hash as incomplete
                # so that next time it will try to recompute the embeddings again
                file_hash = "embeddings_error"

        # Bulk insert into Milvus
        if data_to_insert:
            milvus_client.insert(collection_name=collection_name, data=data_to_insert)

        # Save the new hashes
        hashes[fname] = file_hash

    # Save new collection hash file
    with open(hash_pth, "w") as f:
        json.dump(hashes, f, indent=2)

    elapsed_time = time.time() - start_time
    print(f"⏱️ Finished processing {collection_name} in {elapsed_time:.2f}s\n")
