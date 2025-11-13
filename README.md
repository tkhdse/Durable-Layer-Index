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

# Project Setup: Initializing the Vector Database
This project's database is built from the massive 22 GB MS MARCO document corpus. Because the raw data and the final database are too large to be stored in Git, each team member must run a one-time initialization script to build the database locally.

This script will parse the raw data, generate vector embeddings for all 3.2 million documents, and store them in a persistent ChromaDB instance located outside of this project repository.

## Step 0: Create parent directory
Create a random "598RAP" folder (could be any name) and within that folder, clone our [repo](https://github.com/tkhdse/Durable-Layer-Index.git)

This is what your setup will look like as you follow the steps:

![Parent Folder Setup](images/image.png)

## Step 1: Download the MS MARCO Dataset
Go to the official MS MARCO Datasets website: (https://microsoft.github.io/msmarco/Datasets.html)

On that page, find the "Document Ranking" section.

Download the msmarco-docs.tsv file (22 GB). This is the file shown in the image below, containing 3.2 million documents.

Place the downloaded msmarco-docs.tsv file inside the root of this "598RAP" (or whatever name you pick) folder. The init_db.py script is configured to look for it there.

## Step 2: Install Python Dependencies
This script requires libraries for database management, data parsing, and embedding generation. Install them using pip:

```Bash

pip install chromadb pandas sentence-transformers torch
```

- chromadb: The vector database client.   

- pandas: Used to efficiently read the 22 GB TSV file in chunks.

- sentence-transformers: The library used to generate the document embeddings.   

- torch: Required by sentence-transformers and enables GPU acceleration.

## Step 3: Run the Initialization Script
With the msmarco-docs.tsv file in your project folder and the libraries installed, run the initialization script from your terminal:

```Bash

python init_db.py
```

IMPORTANT: READ THIS
This is a one-time, very long process. The script will parse all 3.2 million documents from the TSV file and generate a vector embedding for each one.

- ON A CPU: This process can take 40+ hours.

- OUTPUT: The script will create a new folder named chroma_persistent_db outside of your Durable-Layered-Index repo folder. This is intentional to keep the large database files (100GB+) out of the Git repository.

- IDEMPOTENT: The script is idempotent. After it finishes successfully, you can run init_db.py again. It will detect the database already exists, skip the 40-hour process, and start instantly.

### Note from Khush: idk what the successful output from the end of the script is supposed to look like so... reach out if you run into any issues. Goal is to have a persistent folder that we can use to create a client when we need to query the VectorDB.