## DB PREPROCESSING DONE ON ICRN CLOUD.




# import chromadb
# import pandas as pd
# import time
# import os
# import csv
# import shutil
# from pathlib import Path
# from sentence_transformers import SentenceTransformer
# import torch

# TSV_FILE_PATH = "./msmarco-docs.tsv" 
# DB_PATH = "./chroma_persistent_db"
# FINAL_ZIP_NAME = "chroma_db_msmarco_complete"

# COLLECTION_NAME = "ms_marco_docs"
# EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# PANDAS_BATCH_SIZE = 4000

# MODEL_ENCODE_BATCH_SIZE = 512

# def _ingest_data(collection: chromadb.Collection, tsv_file_path: str):
#     """
#     Private helper to read the TSV, embed on GPU, and ingest in batches.
#     """
#     print(f"Starting data ingestion for '{collection.name}' from '{tsv_file_path}'...")
    
#     if not os.path.exists(tsv_file_path):
#         print(f"Error: TSV file not found at '{tsv_file_path}'.")
#         return

#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     print(f"Loading model '{EMBEDDING_MODEL_NAME}' onto device: {device}")
#     if device == 'cpu':
#         print("WARNING: CUDA not available. This will be very slow.")
        
#     model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
#     print("Model loaded successfully.")

#     try:
#         chunk_iter = pd.read_csv(
#             tsv_file_path,
#             sep='\t',
#             names=["docid", "url", "title", "body"],
#             header=None,
#             chunksize=PANDAS_BATCH_SIZE, 
#             quoting=csv.QUOTE_NONE 
#         )
#     except Exception as e:
#         print(f"Error opening or reading TSV file: {e}")
#         return

#     print("TSV file opened. Starting batch ingestion...")
#     total_added = 0
#     start_time = time.time()

#     try:
#         for chunk in chunk_iter:
#             chunk = chunk.dropna(subset=["docid", "body"])
#             if chunk.empty:
#                 continue

#             batch_documents = (chunk["title"].astype(str) + "\n\n" + chunk["body"].astype(str)).tolist()
#             batch_ids = chunk["docid"].astype(str).tolist()
#             batch_metadatas = chunk[["url", "title"]].to_dict('records')

#             batch_embeddings = model.encode(
#                 batch_documents, 
#                 batch_size=MODEL_ENCODE_BATCH_SIZE, 
#                 show_progress_bar=False,
#                 convert_to_numpy=True
#             ).tolist() 

#             collection.add(
#                 embeddings=batch_embeddings,
#                 metadatas=batch_metadatas,
#                 ids=batch_ids,
#                 documents=batch_documents
#             )
            
#             total_added += len(batch_ids)
#             elapsed = time.time() - start_time
#             rate = total_added / elapsed if elapsed > 0 else 0
#             print(f"\rAdded {total_added} documents... (Rate: {rate:.2f} doc/sec)", end="")

#         print("\n--- Ingestion Complete ---")
#         print(f"Successfully added {total_added} documents to collection '{collection.name}'.")

#     except Exception as e:
#         print(f"\nAn error occurred during ingestion: {e}")
#     finally:
#         print(f"Final collection count: {collection.count()}")

# def get_or_initialize_db_client(db_path: str):
#     db_path_resolved = str(Path(db_path).resolve())
#     print(f"Initializing persistent client at: {db_path_resolved}")
    
#     client = chromadb.PersistentClient(path=db_path_resolved)

#     print(f"Getting or creating collection: '{COLLECTION_NAME}'")
    
#     collection = client.get_or_create_collection(
#         name=COLLECTION_NAME,
#         metadata={"hnsw:space": "cosine"}
#     )

#     try:
#         count = collection.count()
#         print(f"Collection '{COLLECTION_NAME}' currently contains {count} documents.")
        
#         if count == 0:
#             print("Database is empty. Starting one-time initialization...")
#             _ingest_data(collection, TSV_FILE_PATH)
#         else:
#             print(f"Database is already initialized with {count} documents. Skipping ingestion.")
            
#     except Exception as e:
#         print(f"Error checking collection count: {e}")

#     print("--- ChromaDB Client is ready. ---")
#     return client

# def main():
#     db_storage_path = DB_PATH
#     print("Running database initialization script...")
    
#     get_or_initialize_db_client(db_storage_path)
    
#     print("\nInitialization script finished.")
    
#     print(f"\nZipping the final database folder '{DB_PATH}'...")
#     try:
#         shutil.make_archive(
#             FINAL_ZIP_NAME,        
#             'zip',                  
#             db_storage_path        
#         )
#         print(f"✅ Success! Created '{FINAL_ZIP_NAME}.zip'.")
#         print("You can now download this single file from the VSCode file explorer.")
#     except Exception as e:
#         print(f"Error zipping database: {e}")

# if __name__ == "__main__":
#     main()