# SHL Assessment Recommender

A conversational AI agent that recommends SHL assessments using hybrid retrieval (BM25 + semantic search) with dialogue state tracking.

## Quick Start

```bash
pip install -r requirements.txt
# Optional: set GEMINI_API_KEY in .env for LLM-powered responses
uvicorn app.main:app --reload
```

Open http://localhost:8000 for the chat UI.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check → `{"status": "healthy"}` |
| POST | `/chat` | Chat endpoint → `{session_id, message}` → `{response, recommendations}` |
| GET | `/` | Frontend chat UI |

## Architecture

```
User → FastAPI (/chat) → Agent (Intent→Policy→Action)
                              ↓
                    HybridRetriever (BM25 + TF-IDF + optional SBERT/Reranker)
                              ↓
                    SHL Catalog (377 assessments)
```

- **BM25**: Lexical keyword matching (exact product names, tech terms)
- **TF-IDF/SBERT**: Semantic similarity (intent, role descriptions)
- **Cross-encoder reranker**: Optional precision layer (set `USE_RERANKER=true`)
- **Dialogue state**: Slot extraction + policy-driven actions (clarify/recommend/refine/compare/refuse)

## Evaluation

```bash
python tests/test_evaluation.py
```

Results: 16/16 tests | Recall@10: 0.75 | MRR: 0.65 | 0 hallucinations | 1ms latency

## Deployment

Deploy to Render: push to GitHub, connect repo, deploy using `render.yaml`.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google Gemini API key for LLM responses | (fallback mode) |
| `USE_SBERT` | Enable sentence-transformers embeddings | `false` |
| `USE_RERANKER` | Enable cross-encoder reranking | `false` |
