import itertools
import json
from datasets import load_dataset
from qdrant_client.http import models
from qdrant_client.http.models import PointStruct
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, HnswConfigDiff, Distance

client = QdrantClient(url="http://localhost:6333")
COLLECTION_NAME = "wikipedia"  

info = client.get_collection(COLLECTION_NAME)
print(info)
print(f"Points count: {info.points_count}")

# points = client.scroll(
#     collection_name=COLLECTION_NAME,
#     with_vectors=True,
#     limit=1
# )

# # Try getting a single point by ID
# try:
#     point = client.retrieve(
#         collection_name=COLLECTION_NAME,
#         ids=[1],  # try with different IDs
#         with_vectors=True
#     )
#     print(point)
# except Exception as e:
#     print(f"Error retrieving point: {e}")

# import requests

# url = "http://localhost:6333/collections/wikipedia/"
# r = requests.get(url)

# print("Status code:", r.status_code)
# print("Raw response:", repr(r.text))  # repr() to see empty or whitespace-only

# if r.status_code == 200 and r.text.strip():
#     data = r.json()
#     print(data)
# else:
#     print("Error: No valid JSON returned from Qdrant")
