import os
import time
import chromadb
from chromadb.utils import embedding_functions
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pathlib import Path


load_dotenv()
COLLECTION_NAME = "squad_docs"
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
)


def generate_answer(query: str, docs: list[str]):
    context = "\n\n".join(
        f"Doc {i+1}:\n{docs[i]}" for i in range(len(docs))
    )
    prompt = f"""
        Use the provided context documents ONLY to answer the following question. 
        If the answer is not contained in the context, state that you cannot find the answer.

        Question: {query}

        Context:
        {context}

        Answer:
    """.strip()

    response = llm.invoke(prompt)
    return response.content, context


def query_and_answer(collection: chromadb.Collection, file_path: str, top_k: int = 5):
    if not os.path.exists(file_path):
        print(f"File does not exist: {file_path}")
        return

    with open(file_path, "r") as f:
        queries = [line.strip() for line in f if line.strip()]
        
    with open("output.txt", "a+") as f:
        start_time = time.time()
        for query in queries:
            results = collection.query(
                query_texts=[query],
                n_results=top_k,
            )
            docs = results["documents"][0]
            answer, context = generate_answer(query, docs)

            f.write(f"Query: {query}\nAnswer: {answer}\n\n")

    print(f"Total Time: {round(time.time() - start_time, 3)} seconds")


def main():
    db_path_resolved = str(Path("../chroma_persistent_db").resolve())
    client = chromadb.PersistentClient(path=db_path_resolved)
    embed_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_func
    )
    query_and_answer(collection, "queries.txt", 5)


if __name__ == "__main__":
    main()
