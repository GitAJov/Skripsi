from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333")

client.update_collection(
    collection_name="wikipedia",
    hnsw_config=models.HnswConfigDiff(
        m=8,
    ),
)

client.update_collection(
    collection_name="wikipedia",
    optimizers_config=models.OptimizersConfigDiff(
        indexing_threshold=10000,
    )
)

client.update_collection(
    collection_name="wikipedia",
    optimizer_config=models.OptimizersConfigDiff(),
)

# client.delete_collection("wikipedia")


# client.update_collection(
#     collection_name="wikipedia",
#     hnsw_config=models.HnswConfigDiff(
#         m=8,                # Reduce connectivity to save RAM
#         ef_construct=64,    # Reduce build complexity
#         max_indexing_threads=2,
#         on_disk=True
#     ),
#     optimizers_config=models.OptimizersConfigDiff(
#         indexing_threshold=2000,       # Index in smaller batches
#         memmap_threshold=1000,         # Force mmap earlier
#         default_segment_number=8,      # Start with more segments to avoid giant merges
#         max_segment_size=500000,       # Limit segment size so merges fit in RAM
#         max_optimization_threads=2
#     )
# )

# client.update_collection(
#     collection_name="wikipedia",
#     optimizers_config=models.OptimizersConfigDiff(
#         indexing_threshold=20000,   # Larger threshold to avoid too-small merges
#         max_segment_size=1000000,   # Allow a bit bigger segments
#         default_segment_number=4    # Fewer starting segments, more stable merges
#     )
# )