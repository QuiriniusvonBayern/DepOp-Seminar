import numpy as np
from itertools import combinations
from typing import List


# ============================================================
# Core cosine similarity (lecture notation: S_c)
# ============================================================

def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    """
    Cosine similarity S_c(x,y) as used throughout the lecture.
    """
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom == 0:
        return 0.0
    return float(np.dot(x, y) / denom)


def _cosine_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity matrix between two sets of vectors.
    A: shape (n_A, d)
    B: shape (n_B, d)
    """
    A_norm = np.linalg.norm(A, axis=1, keepdims=True)
    B_norm = np.linalg.norm(B, axis=1, keepdims=True)

    # avoid division by zero
    A_norm[A_norm == 0] = 1.0
    B_norm[B_norm == 0] = 1.0

    return (A @ B.T) / (A_norm @ B_norm.T)


# ============================================================
# Proximity operators (lecture-consistent)
# ============================================================

def proximity_max(set1: np.ndarray, set2: np.ndarray) -> float:
    """
    ProximityMax(S1,S2) = max_{i,j} S_c(v_i, v_j)
    """
    S = _cosine_matrix(set1, set2)
    return float(np.max(S))


def proximity_avg(set1: np.ndarray, set2: np.ndarray) -> float:
    """
    ProximityAvg(S1,S2) = S_c(centroid(S1), centroid(S2))
    """
    c1 = np.mean(set1, axis=0)
    c2 = np.mean(set2, axis=0)
    return cosine_similarity(c1, c2)


def proximity_avg_all(set1: np.ndarray, set2: np.ndarray) -> float:
    """
    ProximityAvgAll(S1,S2) = average of all pairwise cosine similarities.
    """
    S = _cosine_matrix(set1, set2)
    return float(np.mean(S))


def proximity_topn_avg(set1: np.ndarray, set2: np.ndarray, n: int) -> float:
    """
    ProximityTopNAvg(S1,S2):
    For each vector in S1, take its best match in S2,
    then average the top-N of these maxima.
    """
    if n <= 0:
        return 0.0

    S = _cosine_matrix(set1, set2)
    row_maxima = np.max(S, axis=1)
    row_maxima_sorted = np.sort(row_maxima)[::-1]

    n_eff = min(n, len(row_maxima_sorted))
    return float(np.mean(row_maxima_sorted[:n_eff]))


def subset_proximity_avg(set1: np.ndarray, set2: np.ndarray, k: int) -> float:
    """
    SubsetProximityAvg(k,S1,S2):
    Build centroids of all k-sized subsets and apply ProximityMax.
    WARNING: combinatorial explosion for large sets.
    """
    if k <= 0:
        return 0.0

    subsets_1 = [
        np.mean(np.stack(c), axis=0)
        for c in combinations(set1, k)
    ]
    subsets_2 = [
        np.mean(np.stack(c), axis=0)
        for c in combinations(set2, k)
    ]

    A = np.stack(subsets_1)
    B = np.stack(subsets_2)

    return proximity_max(A, B)


# ============================================================
# Weighted cosine similarity (lecture-consistent)
# ============================================================

def weighted_cosine_similarity_rows(
    row_a: List[np.ndarray],
    row_b: List[np.ndarray],
    weights: np.ndarray
) -> float:
    """
    Lecture-consistent weighted cosine similarity:

    S_{c,w}(row_a, row_b)
      = sum_i w_i * S_c(row_{a,i}, row_{b,i}) / sum_i w_i

    - weights correspond to FEATURES (not embedding dimensions)
    - w_i >= 0
    """
    if len(row_a) != len(row_b):
        raise ValueError("row_a and row_b must have the same number of features")

    if len(row_a) != len(weights):
        raise ValueError("Weight vector length must equal number of features")

    weights = np.asarray(weights, dtype=float)

    if np.any(weights < 0):
        raise ValueError("Weights must be non-negative")

    weight_sum = np.sum(weights)
    if weight_sum == 0:
        return 0.0

    sim = 0.0
    for i, w in enumerate(weights):
        if w == 0:
            continue
        sim += w * cosine_similarity(row_a[i], row_b[i])

    return float(sim / weight_sum)


def weighted_proximity_avg(
    row_a: List[np.ndarray],
    row_b: List[np.ndarray],
    weights: np.ndarray
) -> float:
    """
    Alias for the lecture-weighted row similarity.
    Kept for naming consistency with other modules.
    """
    return weighted_cosine_similarity_rows(row_a, row_b, weights)


# ============================================================
# Extensions (NOT part of the lecture core)
# ============================================================

def weighted_proximity_max(row_a, row_b, weights):
    """
    Extension: max over weighted feature-wise cosine similarities.
    NOT a lecture operator.
    """
    sims = [
        weights[i] * cosine_similarity(row_a[i], row_b[i])
        for i in range(len(weights))
        if weights[i] > 0
    ]
    return max(sims) if sims else 0.0


def weighted_proximity_topn_avg(row_a, row_b, weights, n):
    """
    Extension: top-N over feature-wise cosine similarities.
    NOT a lecture operator.
    """
    sims = [
        cosine_similarity(row_a[i], row_b[i])
        for i in range(len(weights))
        if weights[i] > 0
    ]
    sims = sorted(sims, reverse=True)
    n_eff = min(n, len(sims))
    return float(np.mean(sims[:n_eff])) if n_eff > 0 else 0.0


def weighted_proximity_avg_all(row_a, row_b, weights):
    """
    Extension: identical to lecture-weighted similarity for aligned features.
    """
    return weighted_cosine_similarity_rows(row_a, row_b, weights)


# ============================================================
# Legacy helper (not lecture-consistent, kept for compatibility)
# ============================================================

def weighted_centroid(vectors: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Compute a weighted centroid (NOT lecture core).
    """
    weights = np.asarray(weights, dtype=float)
    if np.sum(weights) == 0:
        return np.mean(vectors, axis=0)
    return np.average(vectors, axis=0, weights=weights)
