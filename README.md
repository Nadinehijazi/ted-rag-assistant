# TED RAG Assistant

A **Retrieval-Augmented Generation (RAG) assistant** that answers user questions using knowledge from **TED Talk transcripts**.

The system retrieves relevant TED Talks from a **vector database (Pinecone)** and uses a **Large Language Model (LLM)** to generate grounded answers based on those talks.

This project demonstrates how **semantic search + LLM reasoning** can be combined to build a knowledge assistant over a real dataset.

---

# Project Overview

Large Language Models often hallucinate when answering questions without reliable sources.  
Retrieval-Augmented Generation (RAG) solves this by retrieving relevant documents and using them as context for the model.

In this project:

1. TED Talk transcripts are embedded into vector representations.
2. The embeddings are stored in **Pinecone vector database**.
3. When a user asks a question:
   - the query is embedded
   - relevant talks are retrieved
   - the LLM generates an answer using those talks as context.

This approach produces **more factual and explainable answers**.

Retrieval-Augmented Generation is a widely used architecture for building knowledge assistants and domain-specific AI systems.  

---

# System Architecture

```
User Question
     │
     ▼
Embedding Model
     │
     ▼
Vector Search (Pinecone)
     │
     ▼
Top-K Relevant TED Talks
     │
     ▼
Prompt Construction
     │
     ▼
LLM Generation
     │
     ▼
Answer Grounded in TED Talks
```

---

# Features

- Semantic search over TED Talk transcripts
- Retrieval-Augmented Generation pipeline
- Vector database with **Pinecone**
- Modern **Next.js frontend**
- API endpoints for querying the assistant
- Dataset ingestion pipeline
- Example queries and testing scripts

---

# Tech Stack

Frontend
- Next.js
- React

Backend
- Next.js API routes

AI / Retrieval
- OpenAI embeddings
- Pinecone vector database

Data
- TED Talks dataset (CSV transcripts)

---

# Project Structure

```
ted-rag-assistant/

app/
  api/
    ask/              API endpoint for user questions

data/
  ted_talks_en.csv    TED Talks dataset

scripts/
  ingest.js           loads dataset into Pinecone
  pinecone_stats.js   checks database statistics

public/
  static assets

README.md
package.json
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/your-username/ted-rag-assistant.git
cd ted-rag-assistant
```

Install dependencies:

```bash
npm install
```

---

# Environment Variables

Create a `.env.local` file:

```
OPENAI_API_KEY=your_key
PINECONE_API_KEY=your_key
PINECONE_ENVIRONMENT=your_env
PINECONE_INDEX_NAME=ted-rag-index
```

---

# Data Ingestion

Before using the assistant, load the TED Talks dataset into the vector database.

Run:

```bash
node scripts/ingest.js
```

This script:

1. Reads the TED Talks dataset
2. Creates embeddings
3. Stores them in Pinecone

---

# Running the Application

Start the development server:

```bash
npm run dev
```

Open in browser:

```
http://localhost:3000
```

---

# Example Queries

You can test the assistant with questions such as:

- What makes a good leader?
- What do TED speakers say about creativity?
- How can failure help people succeed?
- What motivates innovation?

The assistant retrieves relevant TED Talks and generates answers grounded in them.

---

# Example Output

Question:

```
What do TED speakers say about creativity?
```

Example Answer:

```
Several TED speakers emphasize that creativity comes from curiosity,
experimentation, and the willingness to make mistakes. For example,
Ken Robinson argues that traditional education systems suppress
creative thinking by prioritizing conformity over exploration.
Other speakers highlight that creativity grows when people connect
ideas across different domains.
```

---


# References

Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks  
https://arxiv.org/abs/2005.11401

Next.js Documentation  
https://nextjs.org/docs

Pinecone Vector Database  
https://www.pinecone.io/docs/
