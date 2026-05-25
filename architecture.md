# System Architecture

The system is strictly divided into an isolated 5-Phase pipeline. This ensures modularity, easy debugging, and swapping of components (e.g., swapping ChromaDB for Pinecone, or LM Studio for an external OpenAI API).

## Technical Stack

- **Backend UI:** Streamlit (app.py)
- **Memory/Vectors:** ChromaDB (database.py)
- **Embeddings:** sentence-transformers/all-MiniLM-L6-v2 (PyTorch)
- **Generative AI Framework:** LangChain (langchain-openai, langchain-core) (agent.py)
- **NLP / RegEx Matching:** spaCy (en_core_web_sm), Python re module (parser.py)

## Flow Diagram

### Phase 1: Ingestion & Parsing (parser.py)

1. **Multithreading:** Triggered by Streamlit, files are sent to a ThreadPoolExecutor to be processed concurrently.
2. **Raw Extraction:** Uses pdfplumber (for PDFs) and docx (for Word arrays) to ingest byte streams without saving files to disk.
3. **NLP Slicing:** A highly optimized spaCy instance (with heavy pipelines disabled like tagger and lemmatizer) extracts candidate names (PERSON entities).
4. **O(1) Master Regex:** A pre-compiled Regex map scans the entire resume array once to find keyword skills (e.g., matching "ML" and "Machine Learning" to machine learning).
5. **Heuristics:** Basic Regex captures Emails, Phone Numbers, and "Years of Experience". Extracts a Pandas DataFrame.

### Phase 2: Logical Chunking & Vector DB (database.py)

1. **Sectioning:** The parser splits resumes into logical blocks (experience, education, skills, projects, etc.).
2. **Embedding:** SentenceTransformer processes these chunks into dense vector embeddings. It attempts to load onto CUDA (GPU), but safely falls back to CPU if PyTorch CUDA isn't supported (as on Python 3.14). Batch chunking (batch_size=32) optimizes speed.
3. **Storage:** Vectors, raw text, and structured Metadata (Years of EXP, is_intern, Skills list) are pushed to ChromaDB.

### Phase 3: Hybrid Ranking Engine (ranker.py)

This calculates a deterministic score to prevent relying entirely on the "black-box" nature of embeddings:

- Rankers extract required skills/experience from the Job Description.
- Pings ChromaDB to pull the top 20 semantic chunks matching the Job Description.
- Calculates Final Score (0.0 - 1.0 scale) using weights:
  - **50% Semantic Score:** Max inverted L2-Distance from the vector database.
  - **30% Skill Overlap:** Strict set intersection of required vs. candidate skills.
  - **20% Experience Fit:** Linear capping of expected vs. provided years of experience.

### Phase 4: Agentic RAG (agent.py)

Utilizes a dynamic 2-step LLM chain architecture connecting to local LM Studio via standard langchain_openai:

1. **Query Router:** Instead of trusting the LLM to write database queries directly, the Router takes natural language ("Find me ML software engineers who are interns") and coerces it into a hard JSON object ({"search_str": "software engineer", "filter": {"is_intern": true}}).
2. **Vector Lookup:** The JSON filter is passed into ChromaDB. The top 5 matching candidate chunks are loaded.
3. **Synthesis Agent:** Forms an augmented prompt with strictly injected text chunks. Applies rules preventing hallucination (e.g., "Cite the candidate_id").

### Phase 5: Dashboard UI (app.py)

Streamlit ties these phases together into interactive tabs, keeping state across re-renders and graphing analytics (e.g., "Missing Skills Distribution" via plotly.express).
