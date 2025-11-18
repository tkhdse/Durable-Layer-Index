import os
import time
import chromadb
from chromadb.utils import embedding_functions
from init_db import get_or_initialize_db_client
from langchain.llms import Ollama
from langchain.smith.evaluation import CriteriaEvalChain

COLLECTION_NAME = "ms_marco_v2_docs"
answer_llm = Ollama(model="llama3", temperature=0.2)


def generate_answer(query: str, docs: list[str]):
    context = "\n\n".join(f"Doc {i+1}:\n{docs[i][:500]}…" for i in range(len(docs)))
    prompt = (
        f" Use the context to answer the question.\n\n"
        f"Question: {query}\n\n"
        f"Context:\n{context}\n\n"
        f"Answer:"
    )
    answer = answer_llm(prompt)
    return answer.strip(), context


def build_evaluator():
    evaluator_llm = Ollama(model="llama3")
    criteria = {
        "factual_accuracy": "Is the answer factually correct based on the provided context?",
        "conciseness": "Is the answer concise and to the point?"
    }
    return CriteriaEvalChain.from_llm(
        llm=evaluator_llm,
        criteria=criteria
    )


def query_and_answer(collection: chromadb.Collection, file_path: str, top_k: int = 5):
    if not os.path.exists(file_path):
        return

    with open(file_path, "r") as f:
        queries = [line.strip() for line in f if line.strip()]

    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction()
    eval_chain = build_evaluator()
    start_time = time.time()
    for query in queries:
        print(f"Query: {query}")
        embedding = embedding_func(query).tolist()
        results = collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents"]
        )
        docs = results["documents"][0]
        answer, context = generate_answer(query, docs)
        print(f"Answer: {answer}")
        eval_result = eval_chain.evaluate(
            input=query,
            prediction=answer,
            reference=context
        )
        print(f"Evaluation: {eval_result}")
        
    print(f"\nTotal Time: {time.time() - start_time}")


def main():
    client = get_or_initialize_db_client()
    collection = client.get_collection(COLLECTION_NAME)
    query_and_answer(collection, "queries.txt", top_k=5)


if __name__ == "__main__":
    main()