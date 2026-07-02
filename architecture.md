# System Architecture

This project follows a simple local pipeline:

`Streamlit UI -> Resume Parser -> ChromaDB -> Ranker / RAG Agent -> Streamlit UI`

## Components

### `app.py`

The Streamlit interface for uploading resumes, entering a job description, viewing rankings, and chatting with the assistant.

### `parser.py`

Reads resume files, extracts text, identifies structured data, and builds a pandas DataFrame for downstream processing.

### `database.py`

Embeds resume sections and stores them in a persistent ChromaDB collection on disk.

### `ranker.py`

Runs deterministic scoring over retrieved candidate chunks to rank resumes against a job description.

### `agent.py`

Uses an OpenAI-compatible local model to:

- turn natural language into a search request
- optionally apply a metadata filter
- synthesize an answer from retrieved resume text

## Data Flow

1. A user uploads one or more resumes.
2. The parser extracts text and metadata.
3. Resume sections are embedded and stored in ChromaDB.
4. The ranker queries the database and computes final candidate scores.
5. The chat assistant performs semantic search and generates a grounded response.

## Design Choices

- Persistent local storage keeps data between app restarts.
- Lazy loading of the embedding model avoids slow startup work until data is actually needed.
- The router is constrained to a small metadata schema to reduce query errors.
- The ranking formula remains deterministic so results are reproducible.

## Extensibility

The code is intentionally split into small modules so each layer can be replaced independently:

- swap ChromaDB for another vector store
- swap LM Studio for another OpenAI-compatible endpoint
- expand the skill map in the parser
- add OCR or better resume normalization later
