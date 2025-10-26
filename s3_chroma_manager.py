import chromadb
import boto3
import os
import shutil
from botocore.exceptions import ClientError
from chromadb.utils import embedding_functions
from typing import Optional

# --- Configuration (UPDATE THESE VALUES) ---
S3_BUCKET_NAME = "your-chroma-vdb-bucket"        # <-- MANDATORY: Replace with your S3 bucket name
S3_KEY_PREFIX = "chroma-data/"                   # Path inside S3 bucket (e.g., staging/vdb/)
LOCAL_PERSIST_DIR = "local_chroma_storage"       # Local working directory (temp folder)
COLLECTION_NAME = "durable_layered_index_v1"
EMBEDDING_MODEL = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2" # Fast, efficient model for local use
)

# --- Boto3 Setup ---
# Boto3 client auto-configures based on environment variables or AWS CLI profile
try:
    s3_client = boto3.client('s3')
except Exception as e:
    print(f"Error initializing Boto3 client: {e}")
    print("Please ensure your AWS credentials are configured correctly.")
    s3_client = None # Placeholder if initialization fails

def get_db_path(local_base_path: str) -> str:
    """Returns the absolute path for the local Chroma persistence directory."""
    return os.path.abspath(local_base_path)

def sync_from_s3(bucket: str, s3_prefix: str, local_path: str):
    """Downloads all ChromaDB files from S3 to the local path."""
    if not s3_client: return 

    print(f"\n--- SYNC INITIATED: Downloading VDB from s3://{bucket}/{s3_prefix} ---")

    if os.path.exists(local_path):
        shutil.rmtree(local_path)
    os.makedirs(local_path, exist_ok=True)
    
    try:
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=s3_prefix)
        
        if 'Contents' not in response or len(response['Contents']) <= 1 and response['Contents'][0]['Key'] == s3_prefix:
            print("S3 data path is empty or does not exist. Starting with a fresh local database.")
            return

        for obj in response.get('Contents', []):
            s3_key = obj['Key']
            # Skip directories or the prefix key itself
            if s3_key.endswith('/') or s3_key == s3_prefix:
                continue

            # Calculate local path
            relative_path = os.path.relpath(s3_key, s3_prefix)
            local_filepath = os.path.join(local_path, relative_path)
            
            os.makedirs(os.path.dirname(local_filepath), exist_ok=True)
            
            print(f"  DOWNLOADING: {s3_key}")
            s3_client.download_file(bucket, s3_key, local_filepath)
        
        print(f"--- Download successful. Files ready in {local_path} ---")

    except ClientError as e:
        print(f"AWS Error during download: {e}")
        raise
    except Exception as e:
        print(f"Error during download: {e}")
        raise

def sync_to_s3(bucket: str, s3_prefix: str, local_path: str):
    """Uploads the entire local ChromaDB directory content to S3."""
    if not s3_client: return 

    print(f"\n--- SYNC INITIATED: Uploading VDB to s3://{bucket}/{s3_prefix} ---")

    if not os.path.exists(local_path):
        print(f"Local directory {local_path} not found. Skipping upload.")
        return
    
    # Walk through the local directory structure
    for root, _, files in os.walk(local_path):
        for file in files:
            local_filepath = os.path.join(root, file)
            # Create the S3 key path relative to the local directory
            relative_path = os.path.relpath(local_filepath, local_path)
            s3_key = os.path.join(s3_prefix, relative_path).replace("\\", "/") # Use forward slash for S3
            
            print(f"  UPLOADING: {local_filepath} -> {s3_key}")
            try:
                s3_client.upload_file(local_filepath, bucket, s3_key)
            except ClientError as e:
                print(f"AWS Error uploading file {local_filepath}: {e}")
                raise
            except Exception as e:
                print(f"Error during upload: {e}")
                raise

    print("--- Upload successful. ChromaDB is now persisted in S3. ---")

def get_chroma_client(local_path: str) -> chromadb.PersistentClient:
    """Initializes and returns the Chroma Persistent Client."""
    print(f"\n--- Initializing ChromaDB PersistentClient at {local_path} ---")
    return chromadb.PersistentClient(path=local_path)

def run_db_operations(client: chromadb.PersistentClient):
    """Performs example ChromaDB operations (indexing, querying)."""
    
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=EMBEDDING_MODEL
    )
    
    # Check if we need to initialize data
    if collection.count() == 0:
        print("Collection is empty. Adding initial documents...")
        collection.add(
            documents=[
                "The Durable Layered Index (DLI) is designed for caching vector queries.",
                "RAG systems combine Large Language Models with external knowledge.",
                "Ollama is used for running large language models locally."
            ],
            metadatas=[
                {"source": "DLI_Spec", "version": "1.0"},
                {"source": "AI_Glossary"},
                {"source": "LLM_Tools"}
            ],
            ids=["doc_dli", "doc_rag", "doc_ollama"]
        )
        print(f"-> Added {collection.count()} documents to the VDB.")
    else:
        print(f"-> Collection '{COLLECTION_NAME}' already has {collection.count()} documents.")
        
    # Example Query
    print("\n--- Performing Query for RAG system component ---")
    results = collection.query(
        query_texts=["What is the purpose of the DLI in this architecture?"],
        n_results=1,
        include=['documents', 'metadatas', 'distances']
    )
    
    print(f"Result Document: {results['documents'][0][0]}")
    print(f"Result Source: {results['metadatas'][0][0]['source']}")
    print("--- Query Complete ---")


def cleanup_local_data(local_path: str):
    """Safely removes the local working directory."""
    try:
        if os.path.exists(local_path):
            shutil.rmtree(local_path)
            print(f"\n--- Cleanup successful: Removed local directory {local_path} ---")
    except Exception as e:
        print(f"Warning: Could not clean up local directory {local_path}: {e}")

# --- MAIN EXECUTION FLOW ---
if __name__ == "__main__":
    
    # 1. Download database from S3
    sync_from_s3(S3_BUCKET_NAME, S3_KEY_PREFIX, LOCAL_PERSIST_DIR)

    # 2. Initialize Chroma Client and perform operations
    chroma_client = get_chroma_client(LOCAL_PERSIST_DIR)
    run_db_operations(chroma_client)

    # 3. Upload the (potentially updated) database back to S3
    sync_to_s3(S3_BUCKET_NAME, S3_KEY_PREFIX, LOCAL_PERSIST_DIR)

    # 4. Cleanup local files (important for serverless or temporary environments)
    # cleanup_local_data(LOCAL_PERSIST_DIR)
    
    print("\n\nS3-BACKED CHROMADB PROCESS FINISHED SUCCESSFULLY.")
