# Durable-Layer-Index
Our solution to the I/O Model Mismatch problem.
Authors: tkhadse2@illinois.edu, uuchei2@illinois.edu, kmaka5@illinois.edu, svbagga2@illinois.edu

## Motivation
Large Language Models (LLMs) such as ChatGPT, Claude, Gemini have revolutionised the way we tackle work; but one long standing problem with them is the accuracy of the information output. Retrieval-Augmented Generation (RAG) is an artificial intelligence technique that combines the text generation of large language models (LLMs) with text retrieval mechanisms. Instead of solely relying on the information that the LLM was trained under when generating text, RAG first retrieves relevant information from external sources (such as documents, APIs, or databases) and then feeds this retrieved content into the language model to generate more accurate and up-to-date responses. 

The disadvantage of the RAG approach is that most databases like S3 are optimised for accessing the entirety of a file, whereas RAG prefers to process files in chunks. Vector databases are ill equipped to handle the random data accesses that a RAG will perform when looking for supplemental data for a query, as a result the queries to a database that RAG makes have significant latency. Our group proposes a Durable Layer Index, which will serve as a layer of abstraction between an LLM and vector database that will reduce the latency the LLM will have with the vector database.

## Implementation
We will create a simple RAG LLM using the Langchain Python Library and Ollama and set up a simple Vector Database using ChromaDB and Boto3. We will then implement our Durable Layered Index as an interface between the RAG LLM and the Vector Database that will reduce the latency of queries the RAG LLM performs on the Vector Database.

The Durable Layered Index will reduce the number of high-latency retrievals the RAG LLM needs to perform when collecting information for its responses.
