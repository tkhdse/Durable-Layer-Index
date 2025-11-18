# Durable-Layer-Index
Our solution to the I/O Model Mismatch problem.

Authors: tkhadse2@illinois.edu, uuchei2@illinois.edu, kmaka5@illinois.edu, svbagga2@illinois.edu


## Background & Motivation
Vector Databases used in systems today generally do not utilize cloud-native support. They are implemented to prioritize write throughput and millisecond-level performance by tightly coupling compute with low-latency storage. For most systems, this approach works, however these systems are less scalable and are not optimized for cost, making it more difficult for a general user to adopt. We explore the efficiency of Cloud-Native Vector Databases, which prioritize cost, durability, and elasticity by decoupling compute from storage while relying on high-latency object storage (like S3). RAG workflows contain stateful, multi-step executions, which involve complex indexing; this impacts the retrieval latency and opens up performance bottlenecks. We aim to address I/O Model Mismatch, which refers to the fundamental conflict between modern indexing structures built on cloud-object storage systems. Our group proposes a Durable Layer Index (DLI), which will serve as an efficient layer of abstraction between an LLM and Vector Database to improve the latency of Cloud-Native Vector Databases.

## Related Works
BinDex: A Two-Layered Index for Fast and Robust Scans
This paper presents a two-layered indexing approach designed for efficient database scan operations, which is closely related to the concept of layered indices optimizing retrieval by structuring access in multiple layers for speed and durability.
RAPTOR RAG: A Hierarchical Indexing for Enhanced Retrieval
Implements recursive summarization and clustering to create a tree-structured multi-layer index for retrieval. Each layer abstracts data. The top layers cover broad topics, while lower layers hold detailed text chunks.
NirDiamant/RAG_Techniques (Miscellaneous Retrieval Techniques GitHub Repo)
An Educational project demonstrating various RAG retrieval strategies including, hierarchical and layered indexing approaches.

## Implementation
We will create a simple RAG LLM using the Langchain Python Library and Ollama and set up a local Vector Database using ChromaDB. We will then use Python to implement our Durable Layered Index (DLI), which will serve an interface between the RAG LLM and the Vector Database that will reduce the latency of queries the RAG LLM performs on the Vector Database. The DLI will additionally be constructed on Redis, which will serve as a semantic cache, and will utilize an LRU eviction policy paired with a TTL per entry. The Durable Layered Index will reduce the number of high-latency retrievals the RAG LLM needs to perform when collecting information for its responses.

## Evaluation
We will evaluate our Durable Layered Index using A/B testing by benchmarking the performance of our LLM with our vector database both with and without the Durable Layered Index. In addition to latency, we will measure end-to-end latency, throughput, and LLM response accuracy of our two setups.

# Project Setup: Initializing the Vector Database and Ollama

Download and extract the contents of the zip file OUTSIDE of this repo folder from this google drive [link](https://drive.google.com/file/d/1Nxdkw4FWXBgtf0HpUV-w8ovjr9GVt_Sr/view?usp=drive_link).

Your local dev setup should look like this (ignore the ms-marco.tsv file and the name should be something other than "chroma_persistent_db" wtv is on the google drive).

![Local Dev Setup](images/image.png)

Following this, download Ollama from the following link [link](https://ollama.com/download). After downloading, run the following commands to get the llama3 model running locally:

```
ollama pull llama3
ollama serve
```