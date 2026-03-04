import numpy as np
from itertools import combinations
from typing import List, Union
import warnings

# ============================================================
# Core cosine similarity
# ============================================================

def cosine_similarity(x: Union[np.ndarray, List], y: Union[np.ndarray, List]) -> float:
    """
    Calculate the cosine similarity between two vectors.

    Args:
        x: First input vector
        y: Second input vector

    Returns:
        Cosine similarity value in range [-1, 1], or 0.0 if either vector has zero norm
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    x_norm = np.linalg.norm(x)
    y_norm = np.linalg.norm(y)
    
    if x_norm == 0 or y_norm == 0:
        return 0.0
    
    return float(np.dot(x, y) / (x_norm * y_norm))


def _pairwise_cosine_matrix(vectors_a: Union[np.ndarray, List], vectors_b: Union[np.ndarray, List]) -> np.ndarray:
    """
    Compute pairwise cosine similarity matrix between two sets of vectors.

    Args:
        vectors_a: Array of shape (n_vectors_a, n_features)
        vectors_b: Array of shape (n_vectors_b, n_features)

    Returns:
        Matrix of shape (n_vectors_a, n_vectors_b) containing pairwise cosine similarities
    """
    vectors_a = np.asarray(vectors_a, dtype=float)
    vectors_b = np.asarray(vectors_b, dtype=float)
    
    if vectors_a.ndim != 2:
        vectors_a = vectors_a.reshape(1, -1) if vectors_a.ndim == 1 else vectors_a.reshape(len(vectors_a), -1)
    if vectors_b.ndim != 2:
        vectors_b = vectors_b.reshape(1, -1) if vectors_b.ndim == 1 else vectors_b.reshape(len(vectors_b), -1)
    
    norms_a = np.linalg.norm(vectors_a, axis=1, keepdims=True)
    norms_b = np.linalg.norm(vectors_b, axis=1, keepdims=True)

    norms_a[norms_a == 0] = 1.0
    norms_b[norms_b == 0] = 1.0

    return (vectors_a @ vectors_b.T) / (norms_a @ norms_b.T)


# ============================================================
# Proximity operators
# ============================================================

def proximity_max(set1: Union[np.ndarray, List], set2: Union[np.ndarray, List]) -> float:
    """
    Calculate maximum pairwise cosine similarity between two sets of vectors.

    Args:
        set1: First set of vectors
        set2: Second set of vectors

    Returns:
        Maximum cosine similarity value across all pairs
    """
    similarity_matrix = _pairwise_cosine_matrix(set1, set2)
    return float(np.max(similarity_matrix))


def proximity_avg(set1: Union[np.ndarray, List], set2: Union[np.ndarray, List]) -> float:
    """
    Calculate cosine similarity between centroids of two vector sets.

    Args:
        set1: First set of vectors
        set2: Second set of vectors

    Returns:
        Cosine similarity of the mean vectors
    """
    set1 = np.asarray(set1, dtype=float)
    set2 = np.asarray(set2, dtype=float)
    
    if set1.ndim == 1:
        set1 = set1.reshape(1, -1)
    if set2.ndim == 1:
        set2 = set2.reshape(1, -1)
    
    centroid1 = np.mean(set1, axis=0)
    centroid2 = np.mean(set2, axis=0)
    return cosine_similarity(centroid1, centroid2)


def proximity_avg_all(set1: Union[np.ndarray, List], set2: Union[np.ndarray, List]) -> float:
    """
    Calculate average of all pairwise cosine similarities between two sets.

    Args:
        set1: First set of vectors
        set2: Second set of vectors

    Returns:
        Mean of all pairwise cosine similarities
    """
    similarity_matrix = _pairwise_cosine_matrix(set1, set2)
    return float(np.mean(similarity_matrix))


def proximity_topn_avg(set1: Union[np.ndarray, List], set2: Union[np.ndarray, List], n: int) -> float:
    """
    Calculate average of top-n best matches from set1 to set2.

    For each vector in set1, find its best match in set2, then average the top-n
    of these maximum similarities.

    Args:
        set1: First set of vectors
        set2: Second set of vectors
        n: Number of top matches to average

    Returns:
        Average of the top-n maximum similarities
    """
    if n <= 0:
        return 0.0
    
    set1 = np.asarray(set1, dtype=float)
    if len(set1) == 0:
        return 0.0
    
    similarity_matrix = _pairwise_cosine_matrix(set1, set2)
    row_maxima = np.max(similarity_matrix, axis=1)
    sorted_maxima = np.sort(row_maxima)[::-1]

    n_effective = min(n, len(sorted_maxima))
    return float(np.mean(sorted_maxima[:n_effective])) if n_effective > 0 else 0.0


def subset_proximity_avg(set1: Union[np.ndarray, List], set2: Union[np.ndarray, List], 
                        subset_size: int, max_combinations: int = 10000) -> float:
    """
    Calculate proximity_max on centroids of all k-sized subsets.

    Args:
        set1: First set of vectors
        set2: Second set of vectors
        subset_size: Size k of subsets to form
        max_combinations: Safety limit to prevent combinatorial explosion

    Returns:
        Maximum cosine similarity between centroids of k-sized subsets
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
    
    from math import comb
    combinations1 = comb(n1, k) if n1 >= k else 0
    combinations2 = comb(n2, k) if n2 >= k else 0
    
    if combinations1 > max_combinations or combinations2 > max_combinations:
        warnings.warn(
            f"Too many combinations (n1={n1}, n2={n2}, k={k}). "
            f"Limited to random sampling.",
            RuntimeWarning
        )
        return _subset_proximity_avg_sampling(set1, set2, k, max_combinations)
    
    if n1 < k or n2 < k:
        return 0.0
    
    subset_centroids1 = []
    subset_centroids2 = []
    
    for combination in combinations(range(n1), k):
        subset = set1[list(combination), :]
        subset_centroids1.append(np.mean(subset, axis=0))
    
    for combination in combinations(range(n2), k):
        subset = set2[list(combination), :]
        subset_centroids2.append(np.mean(subset, axis=0))
    
    if not subset_centroids1 or not subset_centroids2:
        return 0.0
    
    centroids1 = np.stack(subset_centroids1)
    centroids2 = np.stack(subset_centroids2)
    
    return proximity_max(centroids1, centroids2)


def _subset_proximity_avg_sampling(set1: np.ndarray, set2: np.ndarray, 
                                  k: int, n_samples: int) -> float:
    """
    Alternative implementation using random sampling for large combinatorial spaces.

    Args:
        set1: First set of vectors
        set2: Second set of vectors
        k: Subset size
        n_samples: Number of random samples to generate

    Returns:
        Estimated maximum cosine similarity between subset centroids
    """
    np.random.seed(42)
    
    n1, n2 = len(set1), len(set2)
    
    if n1 < k or n2 < k:
        return 0.0
    
    subset_centroids1 = []
    subset_centroids2 = []
    
    for _ in range(min(n_samples, n1)):
        indices = np.random.choice(n1, k, replace=False)
        subset_centroids1.append(np.mean(set1[indices], axis=0))
    
    for _ in range(min(n_samples, n2)):
        indices = np.random.choice(n2, k, replace=False)
        subset_centroids2.append(np.mean(set2[indices], axis=0))
    
    if not subset_centroids1 or not subset_centroids2:
        return 0.0
    
    centroids1 = np.stack(subset_centroids1)
    centroids2 = np.stack(subset_centroids2)
    
    return proximity_max(centroids1, centroids2)


# ============================================================
# Weighted cosine similarity
# ============================================================

def weighted_proximity_avg(
    features_a: List[Union[np.ndarray, List]],
    features_b: List[Union[np.ndarray, List]],
    weights: Union[np.ndarray, List]
) -> float:
    """
    Calculate weighted average of cosine similarities across feature pairs.

    Args:
        features_a: List of feature vectors for first sample
        features_b: List of feature vectors for second sample
        weights: Weight vector for each feature pair

    Returns:
        Weighted average of cosine similarities, or 0.0 if total weight is zero

    Raises:
        ValueError: If input lengths mismatch or weights are negative
    """
    if len(features_a) != len(features_b):
        raise ValueError("features_a and features_b must have the same number of features")
    
    if len(features_a) != len(weights):
        raise ValueError("Weight vector length must equal number of features")
    
    weights = np.asarray(weights, dtype=float)
    
    if np.any(weights < 0):
        raise ValueError("Weights must be non-negative")
    
    total_weight = np.sum(weights)
    if total_weight == 0:
        return 0.0
    
    features_a_arrays = [np.asarray(vec, dtype=float) for vec in features_a]
    features_b_arrays = [np.asarray(vec, dtype=float) for vec in features_b]
    
    weighted_sum = 0.0
    for index, weight in enumerate(weights):
        if weight == 0:
            continue
        weighted_sum += weight * cosine_similarity(features_a_arrays[index], features_b_arrays[index])
    
    return float(weighted_sum / total_weight)
