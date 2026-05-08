import os

from qdrant_client import QdrantClient

QDRANT_MAIN_URL = os.environ.get("QDRANT_URL")
QDRANT_NODES = (
    # swap these...
    "https://node-0-4fedaec2-92e5-4c8b-af38-854cb22c7410.us-east-1-1.aws.cloud.qdrant.io:6333",
    "https://node-1-4fedaec2-92e5-4c8b-af38-854cb22c7410.us-east-1-1.aws.cloud.qdrant.io:6333",
    "https://node-2-4fedaec2-92e5-4c8b-af38-854cb22c7410.us-east-1-1.aws.cloud.qdrant.io:6333",
)
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")

client = QdrantClient(QDRANT_MAIN_URL, api_key=QDRANT_API_KEY)

snapshot_urls = []
for node_url in QDRANT_NODES:
    node_client = QdrantClient(node_url, api_key=QDRANT_API_KEY)
    node_client.create_snapshot(
        collection_name="finewiki-gte-multilingual-base-en", wait=False
    )
