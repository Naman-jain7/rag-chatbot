# DocChat

A Streamlit document-chat application backed by a FastAPI RAG service.

## Run locally

Install dependencies:

```powershell
uv sync
```

Start the FastAPI service:

```powershell
uv run uvicorn app.main:app --reload --port 8000
```

In another terminal, start the Streamlit frontend:

```powershell
uv run streamlit run streamlit_app.py
```

Streamlit connects to `http://127.0.0.1:8000/api/v1` by default. Set
`DOCCHAT_API_URL` before launch to point it at another API URL.

Created by: Naman Jain
