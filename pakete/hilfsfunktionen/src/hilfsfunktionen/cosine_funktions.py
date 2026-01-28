import numpy as np
from itertools import combinations
from typing import List, Union, Optional
import warnings

# ============================================================
# Core cosine similarity (lecture notation: S_c)
# ============================================================

def cosine_similarity(x: Union[np.ndarray, List], y: Union[np.ndarray, List]) -> float:
    """
    Cosine similarity S_c(x,y) as used throughout the lecture.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    x_norm = np.linalg.norm(x)
    y_norm = np.linalg.norm(y)
    
    if x_norm == 0 or y_norm == 0:
        return 0.0
    
    return float(np.dot(x, y) / (x_norm * y_norm))


def _cosine_matrix(A: Union[np.ndarray, List], B: Union[np.ndarray, List]) -> np.ndarray:
    """
    Compute pairwise cosine similarity matrix between two sets of vectors.
    A: shape (n_A, d)
    B: shape (n_B, d)
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    
    # Check dimensions
    if A.ndim != 2:
        A = A.reshape(1, -1) if A.ndim == 1 else A.reshape(len(A), -1)
    if B.ndim != 2:
        B = B.reshape(1, -1) if B.ndim == 1 else B.reshape(len(B), -1)
    
    A_norm = np.linalg.norm(A, axis=1, keepdims=True)
    B_norm = np.linalg.norm(B, axis=1, keepdims=True)

    # avoid division by zero
    A_norm[A_norm == 0] = 1.0
    B_norm[B_norm == 0] = 1.0

    return (A @ B.T) / (A_norm @ B_norm.T)


# ============================================================
# Proximity operators (lecture-consistent)
# ============================================================

def proximity_max(set1: Union[np.ndarray, List], set2: Union[np.ndarray, List]) -> float:
    """
    ProximityMax(S1,S2) = max_{i,j} S_c(v_i, v_j)
    """
    S = _cosine_matrix(set1, set2)
    return float(np.max(S))


def proximity_avg(set1: Union[np.ndarray, List], set2: Union[np.ndarray, List]) -> float:
    """
    ProximityAvg(S1,S2) = S_c(centroid(S1), centroid(S2))
    """
    set1 = np.asarray(set1, dtype=float)
    set2 = np.asarray(set2, dtype=float)
    
    # Reshape if needed
    if set1.ndim == 1:
        set1 = set1.reshape(1, -1)
    if set2.ndim == 1:
        set2 = set2.reshape(1, -1)
    
    c1 = np.mean(set1, axis=0)
    c2 = np.mean(set2, axis=0)
    return cosine_similarity(c1, c2)


def proximity_avg_all(set1: Union[np.ndarray, List], set2: Union[np.ndarray, List]) -> float:
    """
    ProximityAvgAll(S1,S2) = average of all pairwise cosine similarities.
    """
    S = _cosine_matrix(set1, set2)
    return float(np.mean(S))


def proximity_topn_avg(set1: Union[np.ndarray, List], set2: Union[np.ndarray, List], n: int) -> float:
    """
    ProximityTopNAvg(S1,S2):
    For each vector in S1, take its best match in S2,
    then average the top-N of these maxima.
    """
    if n <= 0:
        return 0.0
    
    set1 = np.asarray(set1, dtype=float)
    if len(set1) == 0:
        return 0.0
    
    S = _cosine_matrix(set1, set2)
    row_maxima = np.max(S, axis=1)
    row_maxima_sorted = np.sort(row_maxima)[::-1]

    n_eff = min(n, len(row_maxima_sorted))
    return float(np.mean(row_maxima_sorted[:n_eff])) if n_eff > 0 else 0.0


def subset_proximity_avg(set1: Union[np.ndarray, List], set2: Union[np.ndarray, List], 
                        subset_size: int, max_combinations: int = 10000) -> float:
    """
    SubsetProximityAvg(k,S1,S2):
    Build centroids of all k-sized subsets and apply ProximityMax.
    
    Args:
        max_combinations: Safety limit to prevent combinatorial explosion
    """
    k = subset_size
    if k <= 0:
        return 0.0
    
    set1 = np.asarray(set1, dtype=float)
    set2 = np.asarray(set2, dtype=float)
    
    if set1.ndim == 1:
        set1 = set1.reshape(1, -1)
    if set2.ndim == 1:
        set2 = set2.reshape(1, -1)
    
    n1, n2 = len(set1), len(set2)
    
    # Check for combinatorial explosion
    from math import comb
    comb1 = comb(n1, k) if n1 >= k else 0
    comb2 = comb(n2, k) if n2 >= k else 0
    
    if comb1 > max_combinations or comb2 > max_combinations:
        warnings.warn(
            f"Too many combinations (n1={n1}, n2={n2}, k={k}). "
            f"Limited to random sampling.",
            RuntimeWarning
        )
        # Alternative: Random sampling statt alle Kombinationen
        return _subset_proximity_avg_sampling(set1, set2, k, max_combinations)
    
    # Original implementation with safety checks
    if n1 < k or n2 < k:
        return 0.0
    
    subsets_1 = []
    subsets_2 = []
    
    # Generate centroids for subsets
    for c in combinations(range(n1), k):
        subset = set1[list(c), :]
        subsets_1.append(np.mean(subset, axis=0))
    
    for c in combinations(range(n2), k):
        subset = set2[list(c), :]
        subsets_2.append(np.mean(subset, axis=0))
    
    if not subsets_1 or not subsets_2:
        return 0.0
    
    A = np.stack(subsets_1)
    B = np.stack(subsets_2)
    
    return proximity_max(A, B)


def _subset_proximity_avg_sampling(set1: np.ndarray, set2: np.ndarray, 
                                  k: int, n_samples: int) -> float:
    """Alternative implementation with random sampling."""
    np.random.seed(42)  # Für Reproduzierbarkeit
    
    n1, n2 = len(set1), len(set2)
    
    if n1 < k or n2 < k:
        return 0.0
    
    subsets_1 = []
    subsets_2 = []
    
    for _ in range(min(n_samples, n1)):
        idx = np.random.choice(n1, k, replace=False)
        subsets_1.append(np.mean(set1[idx], axis=0))
    
    for _ in range(min(n_samples, n2)):
        idx = np.random.choice(n2, k, replace=False)
        subsets_2.append(np.mean(set2[idx], axis=0))
    
    if not subsets_1 or not subsets_2:
        return 0.0
    
    A = np.stack(subsets_1)
    B = np.stack(subsets_2)
    
    return proximity_max(A, B)


# ============================================================
# Weighted cosine similarity (lecture-consistent)
# ============================================================

def weighted_proximity_avg(
    row_a: List[Union[np.ndarray, List]],
    row_b: List[Union[np.ndarray, List]],
    weights: Union[np.ndarray, List]
) -> float:
    """
    Lecture-consistent weighted cosine similarity:
    S_{c,w}(row_a, row_b) = sum_i w_i * S_c(row_{a,i}, row_{b,i}) / sum_i w_i
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
    
    # Konvertiere alle Inputs zu Arrays
    row_a_arrays = [np.asarray(vec, dtype=float) for vec in row_a]
    row_b_arrays = [np.asarray(vec, dtype=float) for vec in row_b]
    
    sim = 0.0
    for i, w in enumerate(weights):
        if w == 0:
            continue
        sim += w * cosine_similarity(row_a_arrays[i], row_b_arrays[i])
    
    return float(sim / weight_sum)


# Der Rest des Codes bleibt ähnlich, aber mit Type-Hints für Union[...]