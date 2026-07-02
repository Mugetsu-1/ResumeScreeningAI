# Technical Documentation

## Overview

This application is a local resume screening workflow built around four core steps:

1. Parse uploaded resumes into structured text and metadata.
2. Embed resume sections into a local vector database.
3. Rank candidates against a job description.
4. Answer recruiter questions from retrieved resume context.

## Parsing Pipeline

`parser.py` handles ingestion for PDF and DOCX files.

- PDF text is extracted with `pdfplumber`.
- DOCX text is extracted with `python-docx`.
- Emails and phone numbers are matched with regex.
- Skills are detected with a compact synonym map and a single compiled regex.
- Candidate names are estimated with spaCy, then a simple heuristic fallback is used if the model is not available.
- Resume sections are split into `contact`, `skills`, `experience`, `education`, `projects`, and `other`.

The parser returns a pandas DataFrame with one row per resume.

## Vector Storage

`database.py` stores embedded text chunks in ChromaDB.

- Each non-empty resume section becomes one chunk.
- Metadata includes `candidate_id`, `email`, `skills`, `years_of_exp`, `is_intern`, and `section`.
- The database uses a persistent local directory named `chroma_db/`.
- Embeddings are loaded lazily so the app can start without immediately downloading the model.

## Ranking Logic

`ranker.py` calculates a weighted candidate score.

- Semantic similarity contributes 50 percent.
- Skill overlap contributes 30 percent.
- Experience fit contributes 20 percent.

The final score is a deterministic blend of those three values, sorted from highest to lowest.

## RAG Chat Flow

`agent.py` implements the recruiter assistant.

1. The router converts a user question into JSON.
2. The router may apply a metadata filter such as `is_intern`.
3. ChromaDB returns the closest matching resume sections.
4. The synthesizer answers using only the retrieved text and candidate IDs.

If the model produces malformed JSON, the router falls back to a direct semantic search query with no filter.

## Runtime Assumptions

- The app expects a local OpenAI-compatible endpoint, such as LM Studio.
- The spaCy model `en_core_web_sm` improves name extraction but is optional.
- The ChromaDB collection persists locally between runs.

## Practical Limitations

- Skill matching depends on the skill map in `parser.py`.
- OCR is not included, so image-only PDFs will not parse well.
- The answer quality of the chat assistant depends on the retrieved chunks and the local model.
- This tool should be treated as an assistive screening aid, not an automated hiring decision maker.
