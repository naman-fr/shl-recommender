"""
Hybrid Retrieval Engine for SHL Assessment Catalog.

Three-stage pipeline:
  1. BM25 lexical retrieval — captures exact keyword matches (assessment names, tech terms)
  2. Semantic embedding retrieval — captures intent/meaning similarity
  3. Cross-encoder reranking — precise relevance scoring on merged candidate set

Design rationale (from Qdrant Hybrid Search guide & reranker literature):
- SHL queries mix exact product names ("Java EE 7"), role descriptions ("mid-level developer"),
  skill names ("data science"), and vague intent ("something for customer service").
- BM25 excels at exact keyword precision; embeddings handle semantic recall.
- Cross-encoder reranker provides the final precision layer, producing a ranked top-10.
- Structured filters (job_level, duration, remote, adaptive, category) are applied as
  hard pre-filters before ranking, ensuring recommendations always match explicit constraints.
"""

import json
import re
import os
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

_CATALOG_PATH = Path(__file__).parent.parent / "data" / "shl_catalog.json"

# Flags for optional heavy models
_USE_SBERT = os.environ.get("USE_SBERT", "false").lower() == "true"
_USE_RERANKER = os.environ.get("USE_RERANKER", "false").lower() == "true"


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer for BM25."""
    return re.findall(r'\w+', text.lower())


class HybridRetriever:
    """
    Hybrid retriever: BM25 + TF-IDF/Embeddings + optional cross-encoder reranking.
    Falls back gracefully if sentence-transformers is not available.
    """

    def __init__(self, catalog_path: str | Path | None = None):
        path = Path(catalog_path) if catalog_path else _CATALOG_PATH
        with open(path, "r", encoding="utf-8") as f:
            self.catalog: list[dict] = json.load(f)

        # Build document corpus for each assessment
        self._docs: list[str] = []
        self._tokenized_docs: list[list[str]] = []
        for item in self.catalog:
            doc = self._build_doc(item)
            self._docs.append(doc)
            self._tokenized_docs.append(_tokenize(doc))

        # Stage 1: BM25 index
        self._bm25 = BM25Okapi(self._tokenized_docs)
        logger.info(f"BM25 index built over {len(self.catalog)} documents")

        # Stage 2: TF-IDF (lightweight fallback for semantic similarity)
        self._tfidf = TfidfVectorizer(
            max_features=5000, stop_words="english",
            ngram_range=(1, 2), sublinear_tf=True,
        )
        self._tfidf_matrix = self._tfidf.fit_transform(self._docs)

        # Stage 2b: Sentence-BERT embeddings (optional, for better semantic search)
        self._sbert_model = None
        self._sbert_embeddings = None
        if _USE_SBERT:
            self._init_sbert()

        # Stage 3: Cross-encoder reranker (optional)
        self._reranker = None
        if _USE_RERANKER:
            self._init_reranker()

        # Name lookup index
        self._name_index: dict[str, int] = {}
        for i, item in enumerate(self.catalog):
            self._name_index[item["name"].lower().strip()] = i

    def _build_doc(self, item: dict) -> str:
        """Build a searchable text document from an assessment item."""
        parts = [
            item.get("name", "") * 2,  # Boost name
            item.get("description", ""),
            " ".join(item.get("keys", [])),
            " ".join(item.get("job_levels", [])),
            " ".join(item.get("languages", [])),
        ]
        return " ".join(parts).lower()

    def _init_sbert(self):
        """Initialize sentence-transformers for semantic embeddings."""
        try:
            from sentence_transformers import SentenceTransformer
            model_name = "all-MiniLM-L6-v2"
            logger.info(f"Loading SBERT model: {model_name}")
            self._sbert_model = SentenceTransformer(model_name)
            self._sbert_embeddings = self._sbert_model.encode(
                self._docs, show_progress_bar=False, normalize_embeddings=True
            )
            logger.info("SBERT embeddings computed")
        except Exception as e:
            logger.warning(f"SBERT init failed: {e}. Using TF-IDF fallback.")
            self._sbert_model = None

    def _init_reranker(self):
        """Initialize cross-encoder reranker."""
        try:
            from sentence_transformers import CrossEncoder
            model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
            logger.info(f"Loading reranker: {model_name}")
            self._reranker = CrossEncoder(model_name, max_length=256)
            logger.info("Reranker loaded")
        except Exception as e:
            logger.warning(f"Reranker init failed: {e}. Skipping reranking stage.")
            self._reranker = None

    # ── Public API ───────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 10,
        job_level: Optional[str] = None,
        language: Optional[str] = None,
        max_duration: Optional[int] = None,
        min_duration: Optional[int] = None,
        remote_only: bool = False,
        adaptive_only: bool = False,
        category: Optional[str] = None,
    ) -> list[dict]:
        """
        Hybrid search: BM25 + semantic + rerank, with structured pre-filters.
        Returns up to `top_k` assessments, ranked by relevance.
        """
        # Pre-filter: apply hard constraints
        candidate_indices = self._apply_filters(
            job_level=job_level, language=language,
            max_duration=max_duration, min_duration=min_duration,
            remote_only=remote_only, adaptive_only=adaptive_only,
            category=category,
        )

        if not candidate_indices:
            return []

        if not query or not query.strip():
            return [self.catalog[i].copy() for i in candidate_indices[:top_k]]

        # Stage 1: BM25 scoring over candidates
        query_tokens = _tokenize(query)
        bm25_scores_full = self._bm25.get_scores(query_tokens)
        bm25_scores = {i: bm25_scores_full[i] for i in candidate_indices}

        # Stage 2: Semantic scoring (SBERT or TF-IDF)
        if self._sbert_model is not None and self._sbert_embeddings is not None:
            query_emb = self._sbert_model.encode([query], normalize_embeddings=True)
            cand_embs = self._sbert_embeddings[candidate_indices]
            semantic_scores_arr = cosine_similarity(query_emb, cand_embs).flatten()
            semantic_scores = {candidate_indices[j]: float(semantic_scores_arr[j])
                               for j in range(len(candidate_indices))}
        else:
            query_vec = self._tfidf.transform([query.lower()])
            cand_matrix = self._tfidf_matrix[candidate_indices]
            tfidf_scores_arr = cosine_similarity(query_vec, cand_matrix).flatten()
            semantic_scores = {candidate_indices[j]: float(tfidf_scores_arr[j])
                               for j in range(len(candidate_indices))}

        # Normalize scores to [0,1] and combine (weighted fusion)
        bm25_vals = np.array([bm25_scores.get(i, 0) for i in candidate_indices])
        sem_vals = np.array([semantic_scores.get(i, 0) for i in candidate_indices])

        bm25_max = bm25_vals.max() if bm25_vals.max() > 0 else 1
        sem_max = sem_vals.max() if sem_vals.max() > 0 else 1
        bm25_norm = bm25_vals / bm25_max
        sem_norm = sem_vals / sem_max

        # Weighted: 40% BM25 + 60% semantic (semantic captures intent better)
        combined = 0.4 * bm25_norm + 0.6 * sem_norm

        # Get top candidates (over-retrieve for reranking)
        rerank_k = min(top_k * 3, len(candidate_indices))
        top_indices = np.argsort(combined)[::-1][:rerank_k]
        top_orig_indices = [candidate_indices[j] for j in top_indices]

        # Stage 3: Cross-encoder reranking (if available)
        if self._reranker is not None and len(top_orig_indices) > 1:
            pairs = [(query, self._docs[i]) for i in top_orig_indices]
            rerank_scores = self._reranker.predict(pairs)
            rerank_order = np.argsort(rerank_scores)[::-1][:top_k]
            final_indices = [top_orig_indices[j] for j in rerank_order]
            final_scores = [float(rerank_scores[j]) for j in rerank_order]
        else:
            final_indices = top_orig_indices[:top_k]
            final_scores = [float(combined[top_indices[j]]) for j in range(len(final_indices))]

        results = []
        for idx, score in zip(final_indices, final_scores):
            item = self.catalog[idx].copy()
            item["_score"] = score
            results.append(item)

        return results

    def get_by_name(self, name: str) -> Optional[dict]:
        """Look up by exact name (case-insensitive), with fuzzy fallback."""
        idx = self._name_index.get(name.lower().strip())
        if idx is not None:
            return self.catalog[idx].copy()
        name_lower = name.lower().strip()
        for item in self.catalog:
            if name_lower in item["name"].lower():
                return item.copy()
        return None

    def get_all_categories(self) -> list[str]:
        cats = set()
        for item in self.catalog:
            cats.update(item.get("keys", []))
        return sorted(cats)

    def get_all_job_levels(self) -> list[str]:
        levels = set()
        for item in self.catalog:
            levels.update(item.get("job_levels", []))
        return sorted(levels)

    def get_catalog_size(self) -> int:
        return len(self.catalog)

    # ── Filters ──────────────────────────────────────────────────────────

    def _apply_filters(self, **kwargs) -> list[int]:
        candidates = list(range(len(self.catalog)))

        if kwargs.get("job_level"):
            jl = kwargs["job_level"].lower().strip()
            candidates = [i for i in candidates
                          if any(jl in l.lower() for l in self.catalog[i].get("job_levels", []))]

        if kwargs.get("language"):
            lang = kwargs["language"].lower().strip()
            candidates = [i for i in candidates
                          if any(lang in l.lower() for l in self.catalog[i].get("languages", []))]

        if kwargs.get("max_duration") is not None:
            candidates = [i for i in candidates
                          if self._parse_dur(self.catalog[i].get("duration", "")) is not None
                          and self._parse_dur(self.catalog[i].get("duration", "")) <= kwargs["max_duration"]]

        if kwargs.get("min_duration") is not None:
            candidates = [i for i in candidates
                          if self._parse_dur(self.catalog[i].get("duration", "")) is not None
                          and self._parse_dur(self.catalog[i].get("duration", "")) >= kwargs["min_duration"]]

        if kwargs.get("remote_only"):
            candidates = [i for i in candidates
                          if self.catalog[i].get("remote", "").lower() == "yes"]

        if kwargs.get("adaptive_only"):
            candidates = [i for i in candidates
                          if self.catalog[i].get("adaptive", "").lower() == "yes"]

        if kwargs.get("category"):
            cat = kwargs["category"].lower().strip()
            candidates = [i for i in candidates
                          if any(cat in k.lower() for k in self.catalog[i].get("keys", []))]

        return candidates

    @staticmethod
    def _parse_dur(s: str) -> Optional[int]:
        if not s:
            return None
        m = re.search(r"(\d+)", str(s))
        return int(m.group(1)) if m else None
