import os
import time
import chromadb
from chromadb.utils import embedding_functions
import openai
from init_db import get_or_initialize_db_client
from dotenv import load_dotenv



COLLECTION_NAME = "ms_marco_v2_docs"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_NAME = "gpt-4o-mini"
openai.api_key = OPENAI_API_KEY



def generate_answer_with_openai(query: str, docs: list[str]) -> str:
    context = "\n\n".join(f"Document {i+1}:\n{doc}" for i, doc in enumerate(docs))
    prompt  = (
        f"Use the following context to answer the question.\n\n"
        f"Question: {query}\n\n"
        f"Context:\n{context}\n\n"
        f"Answer:"
    )
    response = openai.ChatCompletion.create(
        model = OPENAI_MODEL_NAME,
        messages = [
            {"role" : "system", "content" : "You are a helpful assistant."},
            {"role" : "user", "content" : prompt}
        ],
        temperature = 0.2
    )
    return response.choices[0].message.content.strip()


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
        res = generate_answer_with_openai(query, docs)
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
