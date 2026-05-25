# AI Resume Screening & Candidate Search System (RAG-Based)

## Overview

This project is a 100% private, locally-hosted AI Resume Screening System. It allows HR professionals or recruiters to upload bulk resumes (PDF/DOCX), parse them intelligently, rank them mathematically against a specific Job Description (JD), and chat with an AI Agent to discover insights from the candidate pool using Retrieval-Augmented Generation (RAG).

Because it connects to local LLM servers (like LM Studio) and relies on an internal ChromaDB vector database, **no candidate data ever leaves your machine**.

## Key Features

- **Phase 1: Concurrent Parsing:** Multithreaded text extraction utilizing pdfplumber and python-docx, combined with stripped-down spaCy NLP and Master Regex algorithms for ultra-fast metadata extraction.
- **Phase 2: Vector Search:** ChromaDB paired with SentenceTransformers (all-MiniLM-L6-v2) converts logical resume segments into searchable semantic vectors. Automatically falls back to CPU due to Python 3.14 PyTorch constraints.
- **Phase 3: Hybrid Mathematical Ranking:** Candidates are ranked against a Job Description using a strict deterministic algorithm balancing semantic similarity, hard skill overlap, and experience fit.
- **Phase 4: Agentic Chatbot:** A two-part LLM agent logic using LangChain. A *Router* converts user intent into DB filters, and a *Synthesizer* enforces strict resume citations to prevent AI hallucinations.
- **Phase 5: Interactive UI:** Built on Streamlit, providing a clean dashboard and Plotly analytics for missed skills.

## Quickstart (Locally)

1. Provide an OpenAI-compatible Local LLM Server (e.g., LM Studio running on port 1234).
2. Activate your virtual environment: .venv\Scripts\Activate.ps1
3. Install requirements (assuming prior execution).
4. Run the UI: streamlit run app.py

## Privacy & Security

This application does not require a cloud connection or paid API keys. Set up LM Studio locally, load an LLM (e.g., Nemotron or Llama3), and the system will proxy its LangChain AI calls directly to your localhost endpoint.
