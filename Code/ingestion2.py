import os
import json
import glob
import pandas as pd
from typing import Iterable
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    PointStruct, VectorParams, HnswConfigDiff,
    Distance, ScalarQuantization, ScalarQuantizationConfig, ScalarType
)

# ==== CONFIG ====
COLLECTION_NAME = "wikipedia"
BATCH_SIZE = 128  # Will be used as the batch_size parameter in upload_points
PROGRESS_FILE = "progress.json"
VECTOR_SIZE = 384
DATA_DIR = r"D:\Life\Academic\Skripsi\Code\data\wikipediasmallLM\data"
NUM_FILES_TO_PROCESS = None # Number of files to process (set to None for all files)
MAX_RETRIES = 5  # Maximum retries for failed batches

# ==== INIT Qdrant ====
client = QdrantClient(url="http://localhost:6333", timeout=60)

def create_collection():
    """Create collection with optimized settings"""
    try:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE,
                on_disk=True
            ),
            hnsw_config=HnswConfigDiff(
                m=0,  # Increased from 0 for better search performance
                ef_construct=100,  # Increased from 50
                on_disk=True
            ),
            optimizers_config=models.OptimizersConfigDiff(
                memmap_threshold=10000,  # Increased from 10000
                indexing_threshold=0,
                max_optimization_threads=4
            ),
            quantization_config=ScalarQuantization(
                scalar=ScalarQuantizationConfig(
                    type=ScalarType.INT8,
                    quantile=0.95,
                    always_ram=False
                )
            )
        )
        print(f"Created collection: {COLLECTION_NAME}")
    except Exception as e:
        print(f"Collection {COLLECTION_NAME} already exists or error: {e}")

create_collection()

# ==== LOAD PROGRESS ====
try:
    with open(PROGRESS_FILE, "r") as f:
        progress_data = json.load(f)
    # Initialize last_file_index if not present
    if "last_file_index" not in progress_data:
        progress_data["last_file_index"] = -1
except FileNotFoundError:
    progress_data = {"last_file_index": -1, "file_progress": {}}

# ==== GET LIST OF FILES ====
files = sorted(glob.glob(os.path.join(DATA_DIR, "*.parquet")))
print(f"Found {len(files)} parquet files.")

# Determine start index
start_index = progress_data["last_file_index"] + 1

# Determine files to process
if NUM_FILES_TO_PROCESS is None:
    end_index = len(files)  # Process all remaining files
else:
    end_index = min(start_index + NUM_FILES_TO_PROCESS, len(files))

if start_index >= end_index:
    print("All files already processed. Nothing to do.")
    exit()

files_to_process = files[start_index:end_index]
print(f"Processing files {start_index} to {end_index-1} ({len(files_to_process)} files)")

def points_generator(df: pd.DataFrame) -> Iterable[PointStruct]:
    """Generator function to yield points one by one"""
    for _, row in df.iterrows():
        yield PointStruct(
            id=int(row["id"]),
            vector=row["emb"].tolist(),  # Ensure vector is list, not numpy array
            payload={
                "title": row["title"],
                "page_content": row["text"]
            }
        )

# ==== PROCESS EACH FILE ====
for file_idx, file_path in enumerate(files_to_process):
    file_name = os.path.basename(file_path)
    global_idx = start_index + file_idx
    
    print(f"\n{'='*50}")
    print(f"Processing file {global_idx+1}/{len(files)}: {file_name}")
    print(f"{'='*50}")
    
    # Load parquet into memory
    df = pd.read_parquet(file_path)

    # Validate data
    df["id"] = df["id"].map(int)
    if not df["id"].is_unique:
        print("Warning: IDs are not unique in this file!")
    
    # Resume position within file
    start_row = progress_data.get("file_progress", {}).get(file_name, 0)
    if start_row > 0:
        print(f"Resuming from row {start_row}")
        df = df.iloc[start_row:]
    
    total_rows = len(df)
    
    # Process using upload_points
    try:
        client.upload_points(
            collection_name=COLLECTION_NAME,
            points=points_generator(df),
            batch_size=BATCH_SIZE,
            max_retries=MAX_RETRIES,
            wait=True,  # Wait for confirmation
        )
        
        print(f"Successfully uploaded {total_rows} points from {file_name}")
        
        # Mark file as complete and update last_file_index
        progress_data["last_file_index"] = global_idx
        if "file_progress" in progress_data and file_name in progress_data["file_progress"]:
            del progress_data["file_progress"][file_name]
        
        with open(PROGRESS_FILE, "w") as f:
            json.dump(progress_data, f)
            
    except Exception as e:
        print(f"Error uploading file {file_name}: {str(e)}")
        # Save progress up to the point where we failed
        with open(PROGRESS_FILE, "w") as f:
            json.dump(progress_data, f)
        continue

print(f"\nProcessed {len(files_to_process)} files. Last file index: {end_index-1}")