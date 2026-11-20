# Comparative RAG: A Practical Comparison of Three Implementations

## Overview

This project explores and compares three different approaches to building a Retrieval-Augmented Generation (RAG) pipeline for question answering over PDF documents.

The same use case is implemented using three approaches:

1. **Basic RAG** – A custom implementation using Python and ChromaDB, without a RAG framework.
2. **LangChain** – An implementation based on LangChain's abstractions and components.
3. **LlamaIndex** – An implementation using LlamaIndex's document processing, indexing and retrieval capabilities.

A shared Gradio interface allows users to select an implementation, ask questions about a PDF document and observe the generated answers.

The objective is to understand how these approaches differ in terms of architecture, abstractions, flexibility and ease of implementation.

## Project Objectives

- Understand the fundamental components of a RAG pipeline.
- Implement the same RAG workflow using three different approaches.
- Compare framework abstractions with a custom implementation.
- Explore document loading, text chunking, embedding generation, vector storage and semantic retrieval.
- Evaluate how each approach integrates with a Large Language Model (LLM).

## RAG Pipeline

Each implementation follows the same general workflow:

1. **Document loading:** Extract text from a PDF document.
2. **Text chunking:** Split the extracted text into smaller segments.
3. **Embedding generation:** Convert text chunks into vector representations.
4. **Indexing:** Store the embeddings and associated text in a vector database.
5. **Retrieval:** Retrieve the most relevant chunks based on the user's question.
6. **Generation:** Provide the retrieved context to an LLM to generate an answer.

## Implementations

### 1. Basic RAG

A custom RAG pipeline implemented with Python and ChromaDB, without relying on a dedicated RAG framework.

This implementation provides direct control over the main pipeline components and serves as a baseline for comparison.

### 2. LangChain

A RAG pipeline built with LangChain.

This implementation explores how LangChain structures the workflow through its abstractions for document processing, embeddings, vector stores and language models.

### 3. LlamaIndex

A RAG pipeline built with LlamaIndex.

This implementation explores LlamaIndex's document-centric architecture, including its document and node abstractions, indexing mechanisms and query engines.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| ChromaDB | Vector database |
| LangChain | RAG framework |
| LlamaIndex | RAG framework |
| Hugging Face | Embedding models |
| Mistral AI | Large Language Model |
| Gradio | User interface |
| Pytest | Testing |

## Project Structure

```text
comparative-rag/
├── src/
│   └── comparative_rag/
│       ├── llm/
│       ├── models/
│       ├── scripts/
│       └── ui/
├── tests/
├── data/
├── app.py
├── pyproject.toml
└── README.md
```

## Running the Project

### 1. Clone the repository

```bash
git clone https://github.com/MlleThalia/DataSciencePortfolio.git
cd comparative-rag
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**macOS / Linux**
```bash
source .venv/bin/activate
```

**Windows**
```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file at the project root and add your Mistral API key:

```env
MISTRALAI_API_URL=https://api.eu.mistral.ai/v1/chat/completions
MISTRALAI_API_KEY=your_api_key
MISTRAL_MODEL=your_model
MAX_RETRY=2
```

### 5. Launch the application

```bash
python -m comparative_rag.app
```

Open the local URL displayed in the terminal to access the Gradio interface.

## Testing

The project includes tests covering the main components of each implementation:

- RAG initialization and configuration.
- Document loading and chunking.
- Document indexing and storage.
- Semantic retrieval.
- Answer generation.

Tests are implemented with Pytest.

To run the test suite:

```bash
pytest -v
```

## Comparison Scope

The project aims to compare implementation approaches rather than establish a definitive ranking of the frameworks.

The comparison focuses on:

- Architecture and design choices.
- Level of abstraction.
- Flexibility and control over pipeline components.
- Developer experience and implementation complexity.
- Integration with embedding models, vector stores and LLMs.

The same PDF question-answering use case is used across implementations to make the comparison more consistent.

## Key Takeaways

This project provides a practical exploration of RAG architectures and highlights the trade-offs between building a pipeline from scratch and using dedicated frameworks.

It also provides a foundation for understanding the internal mechanisms behind higher-level RAG abstractions.