"""
Utilities to view and modify metadata and properties of Milvus collections.
"""

import argparse
import json
import os
from typing import Any

from dotenv import load_dotenv
from pymilvus import MilvusClient

load_dotenv()


def get_milvus_client(
    uri: str | None = None,
    user: str = "root",
    password: str | None = None,
    token: str | None = None,
) -> MilvusClient:
    """
    Get an initialized MilvusClient instance.
    Defaults to environment variables MILVUS_URI and MILVUS_PWD.
    """
    uri = uri or os.getenv("MILVUS_URI")
    password = password or os.getenv("MILVUS_PWD")

    if not uri:
        raise ValueError("MILVUS_URI is not set and was not provided.")

    if token:
        return MilvusClient(uri=uri, token=token)
    return MilvusClient(uri=uri, user=user, password=password)


def get_collection_metadata(
    collection_name: str,
    client: MilvusClient | None = None,
) -> dict[str, Any]:
    """
    Retrieve full metadata for a given Milvus collection, including schema,
    properties, and collection statistics.

    Args:
        collection_name: Name of the collection.
        client: Optional MilvusClient instance.

    Returns:
        Dictionary containing collection metadata and properties.
    """
    if client is None:
        client = get_milvus_client()

    description = client.describe_collection(collection_name=collection_name)
    try:
        stats = client.get_collection_stats(collection_name=collection_name)
        description["stats"] = stats
    except Exception:
        pass

    return description


def get_collection_properties(
    collection_name: str,
    client: MilvusClient | None = None,
) -> dict[str, Any]:
    """
    Retrieve only key-value custom properties of a collection.

    Args:
        collection_name: Name of the collection.
        client: Optional MilvusClient instance.

    Returns:
        Dictionary of custom collection properties.
    """
    metadata = get_collection_metadata(collection_name, client=client)
    return metadata.get("properties", {})


def show_collection_metadata(
    collection_name: str,
    client: MilvusClient | None = None,
    properties_only: bool = False,
) -> dict[str, Any]:
    """
    Display formatted metadata or properties of a Milvus collection.

    Args:
        collection_name: Name of the collection.
        client: Optional MilvusClient instance.
        properties_only: If True, only display custom properties.

    Returns:
        Dictionary of the retrieved metadata or properties.
    """
    metadata = get_collection_metadata(collection_name, client=client)

    if properties_only:
        props = metadata.get("properties", {})
        print(f"\n--- Properties for collection '{collection_name}' ---")
        if props:
            print(json.dumps(props, indent=2))
        else:
            print("No custom properties found.")
        return props

    print(f"\n--- Metadata for collection '{collection_name}' ---")
    print(f"Collection Name:       {metadata.get('collection_name')}")
    print(f"Description:           {metadata.get('description', '')}")
    print(f"Auto ID:               {metadata.get('auto_id')}")
    print(f"Shards:                {metadata.get('num_shards')}")
    print(f"Partitions:            {metadata.get('num_partitions')}")
    print(f"Dynamic Field:         {metadata.get('enable_dynamic_field')}")
    print(f"Consistency Level:     {metadata.get('consistency_level_name')}")
    print(f"Properties:            {json.dumps(metadata.get('properties', {}), indent=2)}")

    fields = metadata.get("fields", [])
    if fields:
        print("\nFields:")
        for f in fields:
            field_name = f.get("name")
            field_type = f.get("type")
            is_pk = f.get("is_primary", False)
            params = f.get("params", {})
            pk_str = " (Primary Key)" if is_pk else ""
            print(f"  - {field_name} [{field_type}]{pk_str}: {params}")

    if "stats" in metadata:
        print(f"\nStats:                 {metadata['stats']}")

    return metadata


def set_collection_properties(
    collection_name: str,
    properties: dict[str, Any],
    client: MilvusClient | None = None,
) -> dict[str, Any]:
    """
    Add or update custom metadata properties for a Milvus collection.

    Args:
        collection_name: Name of the collection.
        properties: Dictionary of key-value properties to set.
        client: Optional MilvusClient instance.

    Returns:
        Dictionary of updated collection properties.
    """
    if client is None:
        client = get_milvus_client()

    # Convert all property values to strings if needed by Milvus
    props_to_set = {k: str(v) if not isinstance(v, (str, int, float, bool)) else v for k, v in properties.items()}

    client.alter_collection_properties(
        collection_name=collection_name,
        properties=props_to_set,
    )
    print(f"✅ Successfully updated properties for '{collection_name}': {props_to_set}")
    return get_collection_properties(collection_name, client=client)


def drop_collection_properties(
    collection_name: str,
    property_keys: list[str] | str,
    client: MilvusClient | None = None,
) -> dict[str, Any]:
    """
    Remove specific property keys from a Milvus collection.

    Args:
        collection_name: Name of the collection.
        property_keys: Single key or list of property keys to drop.
        client: Optional MilvusClient instance.

    Returns:
        Dictionary of remaining collection properties.
    """
    if client is None:
        client = get_milvus_client()

    keys = [property_keys] if isinstance(property_keys, str) else list(property_keys)

    client.drop_collection_properties(
        collection_name=collection_name,
        property_keys=keys,
    )
    print(f"✅ Successfully dropped properties {keys} from '{collection_name}'")
    return get_collection_properties(collection_name, client=client)


# Aliases for convenience
change_collection_metadata = set_collection_properties
update_collection_metadata = set_collection_properties


def main():
    parser = argparse.ArgumentParser(
        description="View and update Milvus collection metadata and properties."
    )
    parser.add_argument(
        "-c", "--collection",
        type=str,
        help="Name of the collection to inspect or modify.",
    )
    parser.add_argument(
        "-l", "--list",
        action="store_true",
        help="List all collections.",
    )
    parser.add_argument(
        "-s", "--show",
        action="store_true",
        help="Show full metadata for the specified collection.",
    )
    parser.add_argument(
        "-p", "--properties",
        action="store_true",
        help="Show only custom properties for the specified collection.",
    )
    parser.add_argument(
        "--set",
        nargs="+",
        metavar="KEY=VALUE",
        help="Set one or more custom properties (e.g. --set owner=public type=test).",
    )
    parser.add_argument(
        "--drop",
        nargs="+",
        metavar="KEY",
        help="Drop one or more custom properties by key (e.g. --drop owner type).",
    )

    args = parser.parse_args()

    client = get_milvus_client()

    if args.list:
        collections = client.list_collections()
        print(f"Found {len(collections)} collections:")
        for col in collections:
            print(f"  - {col}")
        if not args.collection and not args.set and not args.drop:
            return

    if not args.collection:
        if not args.list:
            parser.print_help()
        return

    if args.set:
        props = {}
        for item in args.set:
            if "=" not in item:
                raise ValueError(f"Property must be in KEY=VALUE format, got: {item}")
            k, v = item.split("=", 1)
            props[k.strip()] = v.strip()
        set_collection_properties(args.collection, props, client=client)

    if args.drop:
        drop_collection_properties(args.collection, args.drop, client=client)

    if args.show or (not args.set and not args.drop and not args.properties):
        show_collection_metadata(args.collection, client=client, properties_only=False)
    elif args.properties:
        show_collection_metadata(args.collection, client=client, properties_only=True)


if __name__ == "__main__":
    main()
