"""
Utilities to inspect and display contents of Milvus collections.
Supports arbitrary collection schemas, dynamic fields, and vector exclusion.
"""

import argparse
import json
import os
from typing import Any

from dotenv import load_dotenv
from pymilvus import DataType, MilvusClient
from pymilvus.exceptions import MilvusException


load_dotenv()

def get_milvus_client(
    uri: str | None = None,
    user: str = "root",
    password: str | None = None,
    token: str | None = None,
) -> MilvusClient:
    uri = uri or os.getenv("MILVUS_URI")
    password = password or os.getenv("MILVUS_PWD")
    if not uri:
        raise ValueError("MILVUS_URI is not set and was not provided.")
    if token:
        return MilvusClient(uri=uri, token=token)
    return MilvusClient(uri=uri, user=user, password=password)


VECTOR_DATA_TYPES = {
    DataType.FLOAT_VECTOR,
    DataType.BINARY_VECTOR,
    getattr(DataType, "FLOAT16_VECTOR", 102),
    getattr(DataType, "BFLOAT16_VECTOR", 103),
    getattr(DataType, "SPARSE_FLOAT_VECTOR", 104),
    100,  # FLOAT_VECTOR int code
    101,  # BINARY_VECTOR int code
}


def get_collection_schema_info(
    collection_name: str,
    client: MilvusClient | None = None,
) -> dict[str, Any]:
    """
    Retrieve schema information for a collection: primary key field, scalar fields,
    vector fields, and dynamic field enablement.
    """
    if client is None:
        client = get_milvus_client()

    desc = client.describe_collection(collection_name=collection_name)
    fields = desc.get("fields", [])
    primary_key_field = None
    scalar_fields = []
    vector_fields = []

    for f in fields:
        field_name = f.get("name")
        field_type = f.get("type")
        is_pk = f.get("is_primary", False)

        if is_pk:
            primary_key_field = field_name

        if field_type in VECTOR_DATA_TYPES or (
            isinstance(field_type, str) and "VECTOR" in field_type.upper()
        ):
            vector_fields.append(field_name)
        else:
            scalar_fields.append(field_name)

    return {
        "primary_key": primary_key_field or "id",
        "fields": fields,
        "scalar_fields": scalar_fields,
        "vector_fields": vector_fields,
        "enable_dynamic_field": desc.get("enable_dynamic_field", False),
    }


def get_collection_count(
    collection_name: str,
    client: MilvusClient | None = None,
) -> int:
    """
    Get the total number of entities in a collection.
    """
    if client is None:
        client = get_milvus_client()

    # Try get_collection_stats first
    try:
        stats = client.get_collection_stats(collection_name=collection_name)
        if isinstance(stats, dict) and "row_count" in stats:
            return int(stats["row_count"])
    except (MilvusException, KeyError, TypeError):
        pass

    # Fallback: query count(*)
    try:
        res = client.query(
            collection_name=collection_name,
            filter="",
            output_fields=["count(*)"],
        )
        if res and isinstance(res[0], dict):
            for k in ["count(*)", "count"]:
                if k in res[0]:
                    return int(res[0][k])
    except (MilvusException, KeyError, IndexError, TypeError):
        pass

    # Secondary fallback: count via primary key query with high limit
    try:
        info = get_collection_schema_info(collection_name, client=client)
        pk = info["primary_key"]
        res = client.query(
            collection_name=collection_name,
            filter="",
            output_fields=[pk],
            limit=16384,
        )
        return len(res)
    except (MilvusException, KeyError, TypeError):
        return 0


def _resolve_output_fields(
    collection_name: str,
    output_fields: list[str] | None = None,
    include_vectors: bool = False,
    client: MilvusClient | None = None,
) -> list[str]:
    """
    Determine which fields to retrieve. By default, retrieves all non-vector fields
    (or dynamic fields) so vector representations don't flood the output.
    """
    if output_fields:
        return output_fields

    info = get_collection_schema_info(collection_name, client=client)
    if include_vectors:
        return ["*"]

    # When include_vectors is False, only request scalar fields
    if info["scalar_fields"]:
        return info["scalar_fields"]
    return ["*"]


def get_row_range(
    collection_name: str,
    start: int = 0,
    end: int = 10,
    output_fields: list[str] | None = None,
    include_vectors: bool = False,
    client: MilvusClient | None = None,
) -> list[dict[str, Any]]:
    """
    Retrieve rows within an index range [start, end) (0-indexed).

    Args:
        collection_name: Name of the collection.
        start: Starting row offset (0-based, inclusive).
        end: Ending row offset (exclusive).
        output_fields: Specific fields to retrieve. If None, auto-detected.
        include_vectors: Whether to include embedding vector fields.
        client: Optional MilvusClient instance.

    Returns:
        List of entity dictionaries.
    """
    if client is None:
        client = get_milvus_client()

    start = max(start, 0)
    if end <= start:
        return []

    limit = end - start
    fields = _resolve_output_fields(
        collection_name=collection_name,
        output_fields=output_fields,
        include_vectors=include_vectors,
        client=client,
    )

    return client.query(
        collection_name=collection_name,
        filter="",
        offset=start,
        limit=limit,
        output_fields=fields,
    )


def head(
    collection_name: str,
    n: int = 5,
    output_fields: list[str] | None = None,
    include_vectors: bool = False,
    client: MilvusClient | None = None,
) -> list[dict[str, Any]]:
    """
    Retrieve the first N rows of a collection.

    Args:
        collection_name: Name of the collection.
        n: Number of rows to retrieve.
        output_fields: Specific fields to retrieve.
        include_vectors: Whether to include vector embeddings.
        client: Optional MilvusClient instance.

    Returns:
        List of the first N entity dictionaries.
    """
    if n <= 0:
        return []
    return get_row_range(
        collection_name=collection_name,
        start=0,
        end=n,
        output_fields=output_fields,
        include_vectors=include_vectors,
        client=client,
    )


def tail(
    collection_name: str,
    n: int = 5,
    output_fields: list[str] | None = None,
    include_vectors: bool = False,
    client: MilvusClient | None = None,
) -> list[dict[str, Any]]:
    """
    Retrieve the last N rows of a collection.

    Args:
        collection_name: Name of the collection.
        n: Number of rows to retrieve.
        output_fields: Specific fields to retrieve.
        include_vectors: Whether to include vector embeddings.
        client: Optional MilvusClient instance.

    Returns:
        List of the last N entity dictionaries.
    """
    if n <= 0:
        return []

    if client is None:
        client = get_milvus_client()

    total_rows = get_collection_count(collection_name, client=client)
    start = max(0, total_rows - n)
    return get_row_range(
        collection_name=collection_name,
        start=start,
        end=total_rows,
        output_fields=output_fields,
        include_vectors=include_vectors,
        client=client,
    )


def get_unique_values(
    collection_name: str,
    column_name: str,
    batch_size: int = 1000,
    limit: int | None = None,
    client: MilvusClient | None = None,
) -> list[Any]:
    """
    Retrieve all unique values of a given column across the collection.
    Handles dynamic and schema-defined columns, streaming over batches if necessary.

    Args:
        collection_name: Name of the collection.
        column_name: The column/field name to find unique values for.
        batch_size: Number of records to fetch per batch.
        limit: Optional maximum number of total records to scan.
        client: Optional MilvusClient instance.

    Returns:
        List of unique values found for the column.
    """
    if client is None:
        client = get_milvus_client()

    unique_vals: set[Any] = set()
    offset = 0

    while True:
        cur_batch_size = batch_size
        if limit is not None:
            remaining = limit - offset
            if remaining <= 0:
                break
            cur_batch_size = min(cur_batch_size, remaining)

        batch = client.query(
            collection_name=collection_name,
            filter="",
            offset=offset,
            limit=cur_batch_size,
            output_fields=[column_name],
        )

        if not batch:
            break

        for row in batch:
            if column_name in row:
                val = row[column_name]
                # Convert unhashable types (e.g. dict or list) into json string for uniqueness
                if isinstance(val, (list, dict)):
                    try:
                        unique_vals.add(json.dumps(val, sort_keys=True))
                    except (TypeError, ValueError):
                        unique_vals.add(str(val))
                else:
                    unique_vals.add(val)

        offset += len(batch)
        if len(batch) < cur_batch_size:
            break

    # Convert sorted list if possible, or return as list
    try:
        return sorted(unique_vals)
    except TypeError:
        return list(unique_vals)


def print_entities(entities: list[dict[str, Any]], title: str = "") -> None:
    """
    Nicely format and print a list of entities.
    """
    if title:
        print(f"\n--- {title} (Count: {len(entities)}) ---")
    if not entities:
        print("No entities found.")
        return

    for idx, entity in enumerate(entities):
        print(f"\n[Row {idx + 1}]")
        for key, val in entity.items():
            # If string is long (like chunked text), truncate preview slightly if necessary or print cleanly
            if isinstance(val, str) and len(val) > 300:
                print(f"  {key}: {val[:300]}... [length: {len(val)}]")
            elif isinstance(val, list) and len(val) > 10 and isinstance(val[0], (float, int)):
                print(f"  {key}: [vector of length {len(val)}]")
            else:
                print(f"  {key}: {val}")


def print_row_range(
    collection_name: str,
    start: int = 0,
    end: int = 10,
    output_fields: list[str] | None = None,
    include_vectors: bool = False,
    client: MilvusClient | None = None,
) -> list[dict[str, Any]]:
    """
    Fetch and print rows in the range [start, end).
    """
    rows = get_row_range(
        collection_name=collection_name,
        start=start,
        end=end,
        output_fields=output_fields,
        include_vectors=include_vectors,
        client=client,
    )
    print_entities(rows, title=f"Rows {start} to {end} of '{collection_name}'")
    return rows


def print_head(
    collection_name: str,
    n: int = 5,
    output_fields: list[str] | None = None,
    include_vectors: bool = False,
    client: MilvusClient | None = None,
) -> list[dict[str, Any]]:
    """
    Fetch and print the first N rows.
    """
    rows = head(
        collection_name=collection_name,
        n=n,
        output_fields=output_fields,
        include_vectors=include_vectors,
        client=client,
    )
    print_entities(rows, title=f"Head (First {n} rows) of '{collection_name}'")
    return rows


def print_tail(
    collection_name: str,
    n: int = 5,
    output_fields: list[str] | None = None,
    include_vectors: bool = False,
    client: MilvusClient | None = None,
) -> list[dict[str, Any]]:
    """
    Fetch and print the last N rows.
    """
    rows = tail(
        collection_name=collection_name,
        n=n,
        output_fields=output_fields,
        include_vectors=include_vectors,
        client=client,
    )
    print_entities(rows, title=f"Tail (Last {n} rows) of '{collection_name}'")
    return rows


def print_unique_values(
    collection_name: str,
    column_name: str,
    limit: int | None = None,
    client: MilvusClient | None = None,
) -> list[Any]:
    """
    Fetch and print all unique values of a given column.
    """
    vals = get_unique_values(
        collection_name=collection_name,
        column_name=column_name,
        limit=limit,
        client=client,
    )
    print(
        f"\n--- Unique values for column '{column_name}' in '{collection_name}' (Count: {len(vals)}) ---"
    )
    for v in vals:
        print(f"  - {v}")
    return vals


def main():
    parser = argparse.ArgumentParser(
        description="Inspect and print contents of Milvus collections."
    )
    parser.add_argument(
        "-c", "--collection",
        type=str,
        required=True,
        help="Name of the collection to inspect.",
    )
    parser.add_argument(
        "--head",
        type=int,
        metavar="N",
        help="Print the first N rows.",
    )
    parser.add_argument(
        "--tail",
        type=int,
        metavar="N",
        help="Print the last N rows.",
    )
    parser.add_argument(
        "--range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Print rows in range [START, END) (0-indexed).",
    )
    parser.add_argument(
        "--unique",
        type=str,
        metavar="COLUMN",
        help="Print all unique values of the specified column.",
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        help="Specific fields to display (default: all scalar/dynamic fields).",
    )
    parser.add_argument(
        "--include-vectors",
        action="store_true",
        help="Include vector embeddings in output (can be verbose).",
    )
    parser.add_argument(
        "--max-scan",
        type=int,
        default=None,
        help="Maximum number of rows to scan when computing unique values.",
    )

    args = parser.parse_args()
    client = get_milvus_client()

    has_action = False

    if args.head is not None:
        has_action = True
        print_head(
            collection_name=args.collection,
            n=args.head,
            output_fields=args.fields,
            include_vectors=args.include_vectors,
            client=client,
        )

    if args.tail is not None:
        has_action = True
        print_tail(
            collection_name=args.collection,
            n=args.tail,
            output_fields=args.fields,
            include_vectors=args.include_vectors,
            client=client,
        )

    if args.range is not None:
        has_action = True
        start, end = args.range
        print_row_range(
            collection_name=args.collection,
            start=start,
            end=end,
            output_fields=args.fields,
            include_vectors=args.include_vectors,
            client=client,
        )

    if args.unique is not None:
        has_action = True
        print_unique_values(
            collection_name=args.collection,
            column_name=args.unique,
            limit=args.max_scan,
            client=client,
        )

    if not has_action:
        # Default action: show head 5
        print_head(
            collection_name=args.collection,
            n=5,
            output_fields=args.fields,
            include_vectors=args.include_vectors,
            client=client,
        )


if __name__ == "__main__":
    main()
