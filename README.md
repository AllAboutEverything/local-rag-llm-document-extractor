# Local Privacy-First RAG System

This repository contains a fully local, GDPR-compliant Retrieval-Augmented Generation (RAG) pipeline designed to extract knowledge from internal software documentation without cloud dependencies or data leaks. 

## Technical Architecture & Stack

The architecture is built entirely on a "Privacy-by-Design" philosophy, optimized to run efficiently on consumer-grade hardware (tested on an NVIDIA RTX 2070 Super).

*   **LLM:** `Qwen3:8B` (8-bit quantization for optimized VRAM utilization and context efficiency)
*   **Inference Engine:** `Ollama` (Local, API-driven model orchestration and token streaming)
*   **Vector Database:** `ChromaDB` (Persistent, lightweight, local vector storage for embedded documents)
*   **Embedding Model:** `qwen3-embedding:8b` (High semantic density and robust cross-lingual text alignment)
*   **Frontend UI:** `Streamlit` (User-centric interface featuring asynchronous token streaming and dynamic $k$-adjustment)

## Core Files

*   `ingest.py`: Handles document parsing, text chunking, embedding generation, and persistent storage inside ChromaDB.
*   `app.py`: The conversational interface that manages the local RAG pipeline (Retrieval -> Augmentation -> Generation) via Ollama and presents source metadata to the user.
