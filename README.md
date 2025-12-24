# TED Talk RAG Assistant

A Retrieval-Augmented Generation (RAG) system that answers questions about TED talks using vector similarity search and LLM-based response generation.

## Files

### `preprocess_data.py`
Data preprocessing pipeline that:
- Loads TED talks CSV dataset
- Chunks transcripts using LangChain's RecursiveCharacterTextSplitter (1024 tokens, 20% overlap)
- Generates embeddings using RPRTHPB-text-embedding-3-small
- Uploads vectors to Pinecone database

### `main.py`
FastAPI backend that:
- Exposes `/api/prompt` endpoint for querying the RAG system
- Exposes `/api/stats` endpoint for configuration details
- Retrieves top-20 relevant chunks from Pinecone
- Generates answers using RPRTHPB-gpt-5-mini
- Returns response with context and augmented prompt

### `index.html`
Web interface that

## RAG Configuration

- **Chunk Size**: 1024 tokens
- **Overlap Ratio**: 0.2 (20%)
- **Top-K Retrieval**: 20 chunks
