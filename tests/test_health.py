import os

from dotenv import load_dotenv
from pymilvus import MilvusClient

# Load environment variables
load_dotenv()


def test_milvus_connection():
    # Attempt to retrieve Milvus connection details from the environment
    milvus_uri = os.getenv("MILVUS_URI")
    milvus_pwd = os.getenv("MILVUS_PWD")
    milvus_token = f"root:{milvus_pwd}"

    print("--- Milvus Connection Test ---")
    print(f"Connecting to Milvus URI: {milvus_uri}")
    if milvus_token:
        # Avoid printing the full token but confirm its presence and length
        print(f"Using Token: Yes (length: {len(milvus_token)})")
    else:
        print("Using Token: No (not set in environment)")

    try:
        # Initialize the Milvus client
        client = MilvusClient(uri=milvus_uri, token=milvus_token)
        print("✅ Successfully initialized MilvusClient!")

        # List all collections
        collections = client.list_collections()
        print(f"📂 Found {len(collections)} collection(s):")
        for col in collections:
            print(f" - {col}")

    except Exception as e:
        print("❌ Failed to connect or list collections from Milvus.")
        print(f"Exception details: {e}")


if __name__ == "__main__":
    test_milvus_connection()
