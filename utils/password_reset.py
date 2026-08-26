"""
Replace the default Milvus password with our own password.
"""

import os

from dotenv import load_dotenv
from pymilvus import MilvusClient

load_dotenv()


# Init Milvus
milvus_uri = os.getenv("MILVUS_URI")
milvus_pwd = os.getenv("MILVUS_PWD")

milvus_client = MilvusClient(milvus_uri, token="root:Milvus")

milvus_client.update_password(
    user_name="root",
    old_password="Milvus",
    new_password=milvus_pwd,
)
