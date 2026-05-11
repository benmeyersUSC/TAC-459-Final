# CHANGE: new module for RAG retrieval over resolved tickets
# Uses sentence-transformers all-MiniLM-L6-v2 for fast semantic similarity search

import numpy as np
from typing import List, Dict, Optional

# Lazy-load the model — first call downloads ~80MB, subsequent calls are instant
_model = None


def _get_model():
    """Load the sentence-transformer model on first use, then cache."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model


def embed_text(text: str) -> np.ndarray:
    model = _get_model()
    return model.encode(text, convert_to_numpy=True, show_progress_bar=False)


def embed_texts(texts: List[str]) -> np.ndarray:
    model = _get_model()
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


def cosine_similarity(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Cosine similarity between a query vector and N candidate vectors.
    Returns an array of N similarity scores."""
    # Normalize to unit vectors so dot product equals cosine similarity
    q_norm = query / (np.linalg.norm(query) + 1e-12)
    c_norms = candidates / (np.linalg.norm(candidates, axis=1, keepdims=True) + 1e-12)
    return c_norms @ q_norm


def find_similar_tickets(new_ticket: str,resolved_log: List[Dict],top_k: int = 3,min_similarity: float = 0.4,) -> List[Dict]:
    # resolved only
    candidates = [e for e in resolved_log if e.get('response')]
    if not candidates:
        return []

    # Use cached embedding if present, otherwise compute on the fly
    candidate_texts = [e['ticket'] for e in candidates]
    candidate_embeddings = []
    for entry in candidates:
        if 'embedding' in entry and entry['embedding'] is not None:
            candidate_embeddings.append(np.array(entry['embedding']))
        else:
            # Fallback: compute embedding for old entries that pre-date RAG
            candidate_embeddings.append(embed_text(entry['ticket']))
    candidate_embeddings = np.vstack(candidate_embeddings)

    # Embed the query and rank
    query_embedding = embed_text(new_ticket)
    similarities = cosine_similarity(query_embedding, candidate_embeddings)

    # Sort by similarity descending, filter by threshold, take top_k
    ranked_indices = np.argsort(-similarities)
    results = []
    for idx in ranked_indices:
        sim = float(similarities[idx])
        if sim < min_similarity:
            break
        entry = dict(candidates[idx])  # shallow copy so we don't mutate the log
        entry['similarity'] = sim
        # Don't return the raw embedding to the LLM
        entry.pop('embedding', None)
        results.append(entry)
        if len(results) >= top_k:
            break
    return results
