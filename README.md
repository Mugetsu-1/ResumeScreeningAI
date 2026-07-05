# AI Resume Screening and Candidate Search

Local Streamlit app for screening resumes, ranking candidates against a job description, and querying the candidate pool with a retrieval-augmented chat assistant.

## What it does

- Upload PDF or DOCX resumes from the sidebar.
- Extract contact info, skills, experience hints, and document sections.
- Store resume sections in a local ChromaDB collection.
- Rank candidates against a job description using semantic similarity, skill overlap, and experience fit.
- Ask natural-language questions about the resume pool through a local OpenAI-compatible model such as LM Studio.

## How it works

1. `parser.py` reads each file and extracts text with `pdfplumber` or `python-docx`.
2. The parser uses regex plus spaCy to identify skills, contact details, and a likely candidate name.
3. `database.py` embeds each resume section with `sentence-transformers` and stores the vectors in local ChromaDB persistence.
4. `ranker.py` compares the job description against retrieved chunks and computes a weighted score.
5. `agent.py` routes a question into a semantic search query plus optional metadata filters, then answers from retrieved context.

## Quickstart

1. Create and activate a virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Install the spaCy model if you want better name extraction:

   ```bash
   python -m spacy download en_core_web_sm
   ```

4. Edit `config.py` to point at your LLM provider, endpoint, key, and model.
5. Start a local or remote LLM provider such as LM Studio, Ollama, or OpenAI.
6. Run the app:

   ```bash
   streamlit run app.py
   ```

## Configuration

The app defaults to:

- ChromaDB persistence at `chroma_db/`
- LLM provider: `openai_compatible`
- Base URL: `http://localhost:1234/v1`
- API key placeholder: `lm-studio`
- Model: `deepseek-coder-33b-instruct`

Primary configuration lives in [`config.py`](/D:/ResumeScreeningAI/config.py). You can also override it with environment variables:

- `LLM_PROVIDER`
- `LLM_BASE_URL`
- `LLM_API_KEY`
- `LLM_MODEL`
- `LLM_TIMEOUT`
- `LLM_TEMPERATURE`
- `LLM_MAX_TOKENS`

See `.env.example` for a sample setup.

## Project Structure

- `app.py` - Streamlit UI
- `parser.py` - Resume text extraction and metadata parsing
- `database.py` - ChromaDB persistence and embedding search
- `ranker.py` - Candidate ranking logic
- `agent.py` - RAG chat assistant
- `documentation.md` - Technical notes
- `architecture.md` - High-level system flow

## Notes and Limitations

- The skill dictionary is intentionally small and easy to extend.
- The parser is heuristic-based, so some resumes will extract better than others.
- If the spaCy model is missing, the app still runs, but name extraction is weaker.
- The embedding model is loaded on first use. The first run may take longer if the model is not cached locally.
- This project is a screening aid, not a final hiring decision engine.

## Data Privacy

All resume storage and search happen locally on your machine. The only external calls are to your configured local LLM server and any one-time model downloads you choose to perform.
