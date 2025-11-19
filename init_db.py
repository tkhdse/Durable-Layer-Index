import chromadb
import pandas as pd
import time
import os
import csv
from pathlib import Path

# --- Configuration ---

# **IMPORTANT**: Set this to the exact path where you downloaded 'msmarco-docs.tsv'
TSV_FILE_PATH = "../msmarco-docs.tsv" 

DB_PATH = "../chroma_persistent_db"
COLLECTION_NAME = "ms_marco_docs"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" # A reliable, local model
BATCH_SIZE = 1000  # Number of documents to process and ingest at a time

def _ingest_data(collection: chromadb.Collection, tsv_file_path: str):
    """
    Private helper function to read the 22GB TSV and ingest it in batches.
    This function is only called if the collection is empty.
    """
    print(f"Starting data ingestion for collection '{collection.name}' from '{tsv_file_path}'...")
    
    if not os.path.exists(tsv_file_path):
        print(f"Error: TSV file not found at '{tsv_file_path}'.")
        print("Please download 'msmarco-docs.tsv' and place it in the correct directory.")
        return

    # We read the 22GB file in chunks using pandas to keep memory usage low.
    # The file has no header, so we assign column names.
    # We use csv.QUOTE_NONE (3) because the TSV body may contain unescaped quotes.
    try:
        chunk_iter = pd.read_csv(
            tsv_file_path,
            sep='\t',
            names=["docid", "url", "title", "body"],
            header=None,
            chunksize=BATCH_SIZE,
            quoting=csv.QUOTE_NONE 
        )
    except Exception as e:
        print(f"Error opening or reading TSV file: {e}")
        return

    print("TSV file opened. Starting batch ingestion...")
    total_added = 0
    start_time = time.time()

    try:
        for chunk in chunk_iter:
            # Clean data: Drop rows with missing docid or body
            chunk = chunk.dropna(subset=["docid", "body"])
            if chunk.empty:
                continue

            # Prepare data for ChromaDB
            # We will embed the title and body together for better context
            batch_documents = (chunk["title"].astype(str) + "\n\n" + chunk["body"].astype(str)).tolist()
            batch_ids = chunk["docid"].astype(str).tolist()
            batch_metadatas = chunk[["url", "title"]].to_dict('records')

            # Add the batch to the collection
            # The collection's embedding_function will automatically
            # convert 'batch_documents' to vectors.
            collection.add(
                documents=batch_documents,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
            
            total_added += len(batch_ids)
            elapsed = time.time() - start_time
            rate = total_added / elapsed if elapsed > 0 else 0
            print(f"Added {total_added} documents... (Rate: {rate:.2f} doc/sec)")

        print("\n--- Ingestion Complete ---")
        print(f"Successfully added {total_added} documents to collection '{collection.name}'.")

    except Exception as e:
        print(f"\nAn error occurred during ingestion: {e}")
        print("Ingestion may be incomplete. Please check the error and retry.")
    finally:
        print(f"Final collection count: {collection.count()}")


def get_or_initialize_db_client(db_path: str = "../chroma_persistent_db"):
    """
    Initializes and returns a persistent ChromaDB client.
    
    If the database collection is empty, this function will trigger the 
    full data ingestion process from the local 'msmarco-docs.tsv' file.
    
    Args:
        db_path (str): The directory to store the persistent database.
        
    Returns:
        chromadb.Client: A persistent ChromaDB client instance.
    """
    db_path_resolved = str(Path(db_path).resolve())
    print(f"Initializing persistent client at: {db_path_resolved}")
    
    # Initialize the client
    client = chromadb.PersistentClient(path=db_path_resolved)

    print(f"Loading embedding model: '{EMBEDDING_MODEL_NAME}'")
    # This time, we MUST define an embedding function
    # because the TSV file only contains text.
    from chromadb.utils import embedding_functions
    
    # --- THIS IS THE FIX ---
    # Removed the stray [1] from the end of this function call.
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )
    # ----------------------

    print(f"Getting or creating collection: '{COLLECTION_NAME}'")
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn 
    )

    # --- Check if ingestion is needed ---
    try:
        count = collection.count()
        print(f"Collection '{COLLECTION_NAME}' currently contains {count} documents.")
        
        if count == 0:
            print("Database is empty. Starting one-time initialization...")
            print("This will parse, embed, and ingest 'msmarco-docs.tsv'.")
            print("This process will take several hours and requires significant disk space.")
            _ingest_data(collection, TSV_FILE_PATH)
        else:
            print(f"Database is already initialized with {count} documents. Skipping ingestion.")
            
    except Exception as e:
        print(f"Error checking collection count: {e}")
        print("Proceeding, but the collection might be in an inconsistent state.")

    print("--- ChromaDB Client is ready. ---")
    return client

if __name__ == "__main__":
    db_storage_path = DB_PATH
    # db_storage_path = str(Path(__file__).parent / "chroma_db_storage")
    
    print("Running database initialization script...")
    
    persistent_client = get_or_initialize_db_client(db_storage_path)
    
    print("\nInitialization script finished.")
    print(f"Client object: {persistent_client}")
    
    try:
        collection = persistent_client.get_collection(COLLECTION_NAME)
        print(f"Successfully retrieved collection '{COLLECTION_NAME}' with {collection.count()} documents.")
    except Exception as e:
        print(f"Could not retrieve collection after initialization: {e}")