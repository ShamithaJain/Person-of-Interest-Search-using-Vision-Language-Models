#  -------------------------------------------------------------------------------------------------
#   Copyright (c) 2016-2025.  SupportVectors AI Lab
#  -------------------------------------------------------------------------------------------------
"""Lightweight post-retrieval re-ranking utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np


@dataclass(frozen=True)
class QueryConstraints:
    gender: str | None  # "female" | "male" | None


def infer_constraints(query: str) -> QueryConstraints:
    q = query.lower()
    female_terms = {"woman", "women", "female", "girl", "lady"}
    male_terms = {"man", "men", "male", "boy", "guy"}

    has_f = any(t in q for t in female_terms)
    has_m = any(t in q for t in male_terms)
    gender = None
    if has_f and not has_m:
        gender = "female"
    elif has_m and not has_f:
        gender = "male"
    return QueryConstraints(gender=gender)


def rerank_by_gender(
    results: List[Tuple[str, float]],
    image_embeddings: np.ndarray,
    gender_prompt_embeddings: np.ndarray,
    prefer: str,
    alpha: float = 0.25,
    margin_threshold: float = 0.01,
    hard_penalty: float = 0.8,
) -> List[Tuple[str, float]]:
    """Re-rank FAISS results by adding a gender consistency bonus.

    - results: list of (img_id, base_score) in the same order as image_embeddings rows.
    - image_embeddings: (N, D) normalized.
    - gender_prompt_embeddings: (2, D) normalized for prompts [female_prompt, male_prompt].
    - prefer: "female" or "male"
    - alpha: weight of the bonus term.
    """
    if image_embeddings.shape[0] != len(results):
        return results

    female_sim = (image_embeddings @ gender_prompt_embeddings[0].reshape(-1, 1)).reshape(-1)
    male_sim = (image_embeddings @ gender_prompt_embeddings[1].reshape(-1, 1)).reshape(-1)
    gender_margin = female_sim - male_sim
    if prefer == "male":
        gender_margin = -gender_margin

    rescored = []
    for (img_id, base), margin in zip(results, gender_margin.tolist()):
        # margin > 0 means "looks more like preferred gender prompt"
        score = float(base + alpha * margin)

        # If the model is confidently the *other* gender, push it down harder.
        if margin < -margin_threshold:
            score -= hard_penalty * (-margin)

        rescored.append((img_id, score))
    rescored.sort(key=lambda x: x[1], reverse=True)
    return rescored

