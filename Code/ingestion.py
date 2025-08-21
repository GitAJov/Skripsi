import itertools
import json
from datasets import load_dataset
from qdrant_client.http import models
from qdrant_client.http.models import PointStruct
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, HnswConfigDiff, Distance

# ==== CONFIG ====
COLLECTION_NAME = "wikipedia"  
BATCH_SIZE = 128          # How many to upload in one batch
CHUNK_SIZE = 10000          # How many to process per run
LOOP_COUNT = 10               # Number of chunks to process
PROGRESS_FILE = "progress.json"
VECTOR_SIZE = 384            # Size of the MiniLM-L6-v2 embeddings

# ==== INIT Qdrant ====
client = QdrantClient(url="http://localhost:6333")

# Create collection if not exists
try:
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
            on_disk=True  # Absolutely critical
        ),
        hnsw_config=HnswConfigDiff(
            m=12,                   # Reduced from 16 to save memory
            ef_construct=150,       # Lower than ideal but necessary
            full_scan_threshold=50000,
            on_disk=True            # Non-negotiable
        ),
        optimizers_config=models.OptimizersConfigDiff(
            memmap_threshold=20000,  # Keep only 20k vectors in RAM
            indexing_threshold=0,    # Force immediate ,indexing
        ),
        quantization_config=models.ScalarQuantization(  # <-- Add this section
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8,  # 8-bit quantization
                quantile=0.99,  # Discard extreme 1% values for better accuracy
                always_ram=False  # Critical for your low-RAM setup
        )
    )
    )
    print(f"Created collection: {COLLECTION_NAME}")
except Exception as e:
    print(f"Collection {COLLECTION_NAME} already exists or error: {e}")

# ==== LOAD PROGRESS ====
try:
    with open(PROGRESS_FILE, "r") as f:
        progress_data = json.load(f)
        start_index = progress_data.get("last_index", 0)
        loops_completed = progress_data.get("loops_completed", 0)
except FileNotFoundError:
    start_index = 0
    loops_completed = 0

print(f"Resuming from index {start_index}, loops completed: {loops_completed}")

# ==== MAIN PROCESSING LOOP ====
for loop in range(loops_completed, loops_completed + LOOP_COUNT):
    print(f"\n{'='*50}")
    print(f"Processing loop {loop+1}/{loops_completed + LOOP_COUNT}")
    print(f"{'='*50}")
    
    loop_start = start_index + (loop - loops_completed) * CHUNK_SIZE
    loop_end = loop_start + CHUNK_SIZE
    
    print(f"Processing rows {loop_start} → {loop_end}")

    # ==== STREAM DATASET ====
    dataset_stream = load_dataset(
        "maloyan/wikipedia-22-12-en-embeddings-all-MiniLM-L6-v2",
        split="train",
        streaming=True
    )
    
    # Skip already processed
    stream_iter = itertools.islice(dataset_stream, loop_start, loop_end)
    
    # ==== UPLOAD IN BATCHES ====
    batch = []
    processed_in_loop = 0
    
    for i, doc in enumerate(stream_iter):
        doc_id = doc["id"]
        title = doc["title"]
        text = doc["text"]
        emb = doc["emb"]
        batch.append(
            PointStruct(
                id=doc_id,
                vector=emb,
                payload={"title": title, "page_content": text}
            )
        )
        
        processed_in_loop = i + 1
        
        if len(batch) >= BATCH_SIZE:
            client.upsert(collection_name=COLLECTION_NAME, points=batch)
            batch = []
            if processed_in_loop % (BATCH_SIZE * 10) == 0:  # Print progress every 10 batches
                print(f"Processed {processed_in_loop} docs in current loop")
    
    # Final batch
    if batch:
        client.upsert(collection_name=COLLECTION_NAME, points=batch)
    
    print(f"Loop {loop+1} completed: Processed {processed_in_loop} documents")
    
    # ==== UPDATE PROGRESS ====
    current_index = loop_start + processed_in_loop
    with open(PROGRESS_FILE, "w") as f:
        json.dump({
            "last_index": current_index,
            "loops_completed": loop + 1
        }, f)
    
    print(f"Progress saved: Last index = {current_index}, Loops completed = {loop+1}")
    
    # Early exit if we processed less than a full chunk
    if processed_in_loop < CHUNK_SIZE:
        print(f"Reached end of dataset after {current_index} records")
        break

print("\nProcessing completed!")