# Multi-Modal Stateful RAG Chatbot

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.2+-FF4B4B.svg)](https://github.com/langchain-ai/langgraph)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-336791.svg)](https://github.com/pgvector/pgvector)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An enterprise-grade, multi-modal Retrieval-Augmented Generation (RAG) system featuring a **5-node stateful LangGraph pipeline**, **hybrid vector + full-text search** in PostgreSQL with cross-encoder reranking, **audio transcription ingestion** via Faster-Whisper, and a **fault-tolerant multi-provider LLM resilience layer** with circuit breakers.

---

## 1. Title & Description

**Multi-Modal Stateful RAG Chatbot** is a production-ready conversational AI platform backed by an asynchronous **FastAPI** backend and an interactive **Streamlit** frontend. 

The system enables users to upload multi-format documents (PDFs, Markdown, HTML, 9 programming languages, plain text) and audio recordings (MP3, WAV, M4A, OGG, FLAC) into an isolated per-user knowledge base. Queries are processed through an adaptive LangGraph decision workflow that intelligently decides between answering from conversational short-term memory or executing a two-stage hybrid retrieval pipeline (HNSW cosine similarity + GIN BM25 full-text search) followed by cross-encoder reranking and automated evidence verification.

---

## 2. Problem Statement

Standard RAG architectures often suffer from key production bottlenecks:
1. **Context Fragmentation & Noise:** Naive top-$k$ vector retrieval frequently pulls irrelevant or poorly ordered chunks into the prompt, degrading answer precision and increasing hallucinations.
2. **Repetitive & Unnecessary Retrievals:** Conversational follow-ups (e.g., *"What was the second point you mentioned?"*) frequently trigger expensive vector lookups even when the context is already present in conversation history.
3. **Single Provider Fragility:** LLM API outages, rate limits, or transient network timeouts break end-to-end user sessions without automated fallback.
4. **Single-Modality Constraints:** Audio files (lectures, voice memos, meetings) are excluded or require external preprocessing pipelines.

This project addresses these challenges with an integrated, stateful orchestration graph, hybrid search with cross-encoder reranking, multi-modal ingestion, and automated LLM circuit breakers.

---

## 3. Dataset

- **Supported Formats:**
  - **Documents & Code:** `.pdf`, `.md`, `.html`, `.txt`, `.py`, `.js`, `.ts`, `.java`, `.cpp`, `.go`, `.rb`, `.php`, `.rs`
  - **Audio Formats:** `.mp3`, `.wav`, `.m4a`, `.ogg`, `.flac`
- **Deduplication:** SHA-256 content hashing prevents duplicate chunk storage per user.

---

## 4. Methodology & Pipeline Architecture

```mermaid
flowchart TD
    A([User Query]) --> B[START: check_short_term]
    
    B -->|Short-term memory sufficient| G[generate_answer]
    B -->|Document query or insufficient context| C[retrieve_docs: Hybrid Search]
    
    subgraph Hybrid Retrieval & Reranking
        C --> D1[pgvector HNSW Cosine Search - Top 20]
        C --> D2[GIN tsvector BM25 Search - Top 20]
        D1 --> D3[Merge & Deduplicate - Up to 40 candidates]
        D2 --> D3
        D3 --> D4[BAAI/bge Cross-Encoder Reranker - Top 5]
    end
    
    D4 --> E[evaluate_evidence]
    
    E -->|Evidence Sufficient or Retries ≥ 1| G
    E -->|Evidence Insufficient & Has Retries left| F[rewrite_query]
    
    F --> C
    
    G --> H([Token Streaming Output & Citations])
```


Retrieval & Decision Flow:
1. Short-Term Context Check: Queries are analyzed using structured LLM classification against recent chat history. If the query references prior dialogue, retrieval is bypassed to minimize latency and token consumption.

2. Parallel Hybrid Search:
    - Vector Search: HNSW index on embeddings using cosine distance (<=>).
    - Keyword Search: PostgreSQL GIN full-text index on tsvector with websearch_to_tsquery.
    - Queries execute concurrently via asyncio.gather, pulling up to 20 candidates each (40 total).

3. Cross-Encoder Reranking: Candidate chunks are scored against the user query using BAAI/bge-reranker-v2-m3 to select the top 5 most relevant passages.

4. Evidence Evaluation & Query Rewriting: A structured evaluator inspects retrieved passages. If evidence is insufficient, the query is rewritten and re-executed (up to 1 retry).

5.  Stateful Streaming & Checkpointing: The answer is streamed back in real time over asyncio.Queue via StreamingResponse. Thread state is persisted in PostgreSQL using AsyncPostgresSaver.

---

## 5. Model Details & Configurations

| Component | Model / Engine | Configuration / Parameters |
| :--- | :--- | :--- |
| **Embeddings** | HuggingFace Sentence-Transformers | 384 / 768 / 1536 dim (configured via `.env`) |
| **Vector Index** | PostgreSQL `pgvector` HNSW | `m = 16`, `ef_construction = 200`, `vector_cosine_ops` |
| **Full-Text Index** | PostgreSQL GIN | `tsvector` generated on `english` dictionary |
| **Cross-Encoder Reranker** | `BAAI/bge-reranker-v2-m3` | Input: `(query, chunk_text)` pairs; output: top-5 |
| **Audio Transcription** | `faster-whisper` (CTranslate2) | Model: `base`, Compute: `int8` / `float16`, `beam_size = 5`, auto-detect CPU/GPU |
| **LLM Provider Stack** | Multi-Provider Cascade | Local Ollama, Cloud Ollama, OpenRouter, Google Gemini |
| **Resilience & Circuit Breaker** | `pybreaker` + `tenacity` | `fail_threshold = 3`, `cooldown = 30s`, exponential backoff |
| **Text Splitter** | Recursive & Header-Aware Splitters | `chunk_size = 1000`, `chunk_overlap = 200` |

---

## 6. Project Structure

```text
rag_chatbot/
├── app/
│   ├── api/
│   │   ├── auth.py             # JWT signup/login & password hashing
│   │   ├── chat.py             # Streaming chat endpoint & conversation history CRUD
│   │   ├── delete_docs.py      # Document chunk & metadata deletion
│   │   ├── ingest_audio.py     # Faster-Whisper audio transcription & ingestion
│   │   ├── ingest_docs.py      # Multi-format doc/code parsing & chunking
│   │   ├── list_docs.py        # Document listing endpoint
│   │   └── router.py           # Central FastAPI route registry
│   ├── core/
│   │   └── config.py           # Pydantic BaseSettings & multi-provider configurations
│   ├── db/
│   │   ├── manager.py          # Database manager singleton instance
│   │   └── postgres_db.py      # Thread-safe asyncpg connection pool wrapper
│   ├── schemas/
│   │   ├── user_profile.py     # Pydantic schemas for authentication
│   │   └── workflow_schema.py  # LangGraph state schemas & structured evaluation
│   └── main.py                 # FastAPI application & lifespan management
├── src/
│   ├── audio/
│   │   ├── chunker.py          # Overlap-aware Whisper segment chunker
│   │   └── transcriber.py      # Lazy-loaded faster-whisper transcription engine
│   ├── graphs/
│   │   ├── chains.py           # 5-node LangGraph workflow definition & routing
│   │   ├── state.py            # GraphState typed dictionary definition
│   │   └── tools.py            # External tools (DuckDuckGo, Stock, Currency, Calc)
│   ├── llm/
│   │   ├── base.py             # Abstract BaseLLMProvider interface
│   │   ├── config.py           # Provider configs & retry parameters
│   │   ├── providers.py        # Ollama, Gemini, OpenRouter implementations
│   │   └── resilience.py       # ResilientLLMManager with circuit breakers & retries
│   ├── memory/
│   │   ├── short_term_memory.py# AsyncPostgresSaver checkpointer builder
│   │   └── vector_db_manager.py# Hybrid retrieval (HNSW + BM25) & cross-encoder
│   ├── prompts/
│   │   └── prompts.py          # System prompts for evaluation, rewrite & generation
│   └── utils/
│       ├── citations.py        # Source citation formatter
│       ├── exception.py        # Custom application exceptions
│       ├── logger.py           # Unified loguru/standard loggers
│       └── token_usage.py      # Token counting & session metrics
├── pages/                      # Streamlit UI sub-pages
├── streamlit_app.py            # Streamlit UI main entry point
├── run_backend.py              # Backend launcher script
├── run_project.py              # Full-stack orchestrator script
├── pyproject.toml              # Project dependencies & package metadata
└── README.md
```

---

## 7. Installation & Setup

### Prerequisites

- Python: 3.11 or 3.12
- Package Manager: uv (recommended) or pip
- Database: PostgreSQL 15+ with the pgvector extension enabled
- Local LLM (Optional): Ollama installed and running locally

1. Clone the Repository

```bash
git clone https://github.com/<your-username>/rag-chatbot.git
cd rag-chatbot
```

2. Set Up Virtual Environment & Dependencies

```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

3. Configure Environment Variables


```env
APP_NAME=RAGChatbot
APP_VERSION=1.0.0
SECRET_KEY=your_secure_random_secret_key
LANGCHAIN_PROJECT=rag_chatbot

DB_DSN=postgresql://postgres:postgres@localhost:5432/rag_chatbot
DB_PATH=data/app.db
VECTOR_DB_PATH=data/vector.db
SYS_PROMPTS_PATH=src/prompts/

HUGGINGFACEHUB_API_TOKEN=your_hf_token
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
EMBEDDINGS_TABLE_NAME=document_chunks
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

WHISPER_MODEL_SIZE=base
WHISPER_COMPUTE_TYPE=int8
WHISPER_DEVICE=auto

OLLAMA_LOCAL_PROVIDER_NAME="ollama local"
OLLAMA_LOCAL_MODEL_NAME=llama3.2
OLLAMA_LOCAL_BASE_URL=http://localhost:11434
OLLAMA_LOCAL_PRIORITY=1

OLLAMA_PROVIDER_NAME=ollama
OLLAMA_MODEL_NAME=llama3.2
OLLAMA_API_KEY=
OLLAMA_PRIORITY=2

OPENROUTER_PROVIDER_NAME=openrouter
OPENROUTER_MODEL_NAME=meta-llama/llama-3.1-8b-instruct:free
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_PRIORITY=3

GEMINI_PROVIDER_NAME=gemini
GEMINI_MODEL_NAME=gemini-1.5-flash
GOOGLE_API_KEY=your_google_api_key
GEMINI_PRIORITY=4

TIMEOUT=30
MAX_RETRIES=2
CIRCUIT_BREAKER_THRESHOLD=3
CIRCUIT_BREAKER_COOLDOWN=30
MAX_TOKENS=1024
TEMPERATURE=0.3

ALPHAVANTAGE_STOCK_API_KEY=your_alphavantage_key
CURRENCY_EXCHANGE_URL=https://v6.exchangerate-api.com/v6
CURRENCY_EXCHANGE_API_KEY=your_exchange_rate_key
```

## 8. Usage Examples

### Running the Application

- Option A: Run Full Stack (Backend + Frontend)

```bash
uv run python run_project.py
```

- Option B: Run Services Individually

1. Start FastAPI Backend:

```bash
uv run python run_backend.py
# API will be available at http://localhost:8000 (Swagger docs at /docs)
```

2. Start Streamlit UI:

```bash
uv run streamlit run streamlit_app.py
# Frontend will open at http://localhost:8501
```

---

## 9. Tech Stack

- **Backend Framework**: FastAPI, Uvicorn, Pydantic v2, Pydantic-Settings
- **Orchestration & Workflow**: LangGraph, LangChain, LangSmith (observability & tracing)
- **Database & Storage**: PostgreSQL, pgvector, asyncpg (app pool), psycopg3 / psycopg-pool (checkpointing)
- **Embedding & ML Models**: HuggingFace sentence-transformers, CrossEncoder (BAAI/bge-reranker-v2-m3), faster-whisper, ctranslate2
- **Fault Tolerance & Reliability**: pybreaker (Circuit Breaker pattern), tenacity (Async retrying with exponential backoff)
- **Frontend UI**: Streamlit
- **Document Parsers**: pypdf, HTMLHeaderTextSplitter, MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
- **Security & Auth**: bcrypt, pyjwt, OAuth2 password bearer

---

## 10. Future Work

-  Agentic Tool Execution in Graph: Incorporate dynamic tool-calling branches directly into the LangGraph state machine for live web search, currency, and financial computations.
- Speaker Diarization: Integrate speaker separation (e.g., PyAnnote) into the audio transcription pipeline to attribute spoken segments to individual speakers.
- Hierarchical & Parent-Document Retrieval: Index small semantic chunks mapped back to larger parent paragraphs for enriched LLM context windows.
 Asynchronous Task Queue: Introduce Celery/Redis background workers for long-running batch ingestion of large video/audio archives.