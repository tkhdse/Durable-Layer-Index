import os
import time
import chromadb
from chromadb.utils import embedding_functions
from init_db import get_or_initialize_db_client
from dotenv import load_dotenv
import requests



COLLECTION_NAME = "ms_marco_docs"



def generate_answer(query: str, docs: list[str]) -> str:
    context = "\n\n".join(f"Doc {i+1}:\n{docs[i][:500]}…" for i in range(len(docs)))
    prompt = (
        f"You are a helpful assistant. Use the context to answer the question.\n\n"
        f"Question: {query}\n\n"
        f"Context:\n{context}\n\n"
        f"Answer:"
    )
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": prompt,
            "temperature": 0.2,
        },
        timeout=300,
    )

    data = response.json()
    return data.get("response", "").strip()


def query_and_answer(collection: chromadb.Collection, file_path: str, top_k: int = 5):
    if not os.path.exists(file_path):
        return
    with open(file_path, "r") as f:
        queries = [line.strip() for line in f if line.strip()]
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction()
    start_time = time.time()
    for query in queries:
        print(f"Query: {query}\n")
        embedding = embedding_func(query).tolist()
        results = collection.query(
            query_embeddings = [embedding],
            n_results = top_k,
            include = ["documents"]
        )
        docs = results["documents"][0]
        res = generate_answer(query, docs)
        print(f"Answer: {res}\n")
    end_time = time.time()
    print(f"Normal RAG Time: {end_time - start_time}\n")


def main():
    load_dotenv()
    client = get_or_initialize_db_client()
    collection = client.get_collection(COLLECTION_NAME)
    query_and_answer(collection, "queries.txt", top_k=5)


if __name__ == "__main__":
    main()
