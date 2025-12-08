import numpy as np
from itertools import combinations
from typing import List, Tuple, Iterable

# -------------------------
# Grundfunktion
# -------------------------
def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    v1 = np.asarray(v1, dtype=float).ravel()
    v2 = np.asarray(v2, dtype=float).ravel()
    if v1.shape != v2.shape:
        raise ValueError(f"Dimensionen stimmen nicht überein: {v1.shape} vs {v2.shape}")
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


# -------------------------
# Hilf: bereite Matrixformen vor
# -------------------------
def _as_matrix(vecs: Iterable[np.ndarray]) -> np.ndarray:
    vecs = list(vecs)
    if not vecs:
        return np.zeros((0, 0), dtype=float)
    mat = np.vstack([np.asarray(v, dtype=float).ravel() for v in vecs])
    return mat  # shape: (m, d)


# -------------------------
# 1) ProximityMax
# -------------------------
def proximity_max(set1: List[np.ndarray], set2: List[np.ndarray]) -> float:
    A = _as_matrix(set1)  # shape (m, d)
    B = _as_matrix(set2)  # shape (n, d)
    if A.size == 0 or B.size == 0:
        return 0.0
    # Dot products
    D = A @ B.T  # shape (m, n)
    normsA = np.linalg.norm(A, axis=1)
    normsB = np.linalg.norm(B, axis=1)
    denom = np.outer(normsA, normsB)  # shape (m, n)
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        S = np.divide(D, denom, out=np.zeros_like(D), where=denom != 0)
    return float(np.max(S))


# -------------------------
# 2) ProximityAvg (Durchschnittsvektoren)
# -------------------------
def proximity_avg(set1: List[np.ndarray], set2: List[np.ndarray]) -> float:
    if not set1 or not set2:
        return 0.0
    A = _as_matrix(set1)
    B = _as_matrix(set2)
    avgA = np.mean(A, axis=0)
    avgB = np.mean(B, axis=0)
    return cosine_similarity(avgA, avgB)


# -------------------------
# 3) ProximityTopNAvg (gemäß Tabelle: pro Element max und dann Top-N)
# -------------------------
def proximity_topn_avg(set1: List[np.ndarray], set2: List[np.ndarray], n: int = 3) -> float:
    A = _as_matrix(set1)
    B = _as_matrix(set2)
    if A.size == 0 or B.size == 0:
        return 0.0
    D = A @ B.T
    normsA = np.linalg.norm(A, axis=1, keepdims=True)  # (m,1)
    normsB = np.linalg.norm(B, axis=1, keepdims=True)  # (n,1)
    denom = normsA @ normsB.T  # (m,n)
    with np.errstate(divide='ignore', invalid='ignore'):
        S = np.divide(D, denom, out=np.zeros_like(D), where=denom != 0)
    # Für jede Zeile (jedes v in set1) das Maximum über Spalten (set2)
    row_max = np.max(S, axis=1)  # shape (m,)
    if row_max.size == 0:
        return 0.0
    row_max_sorted = np.sort(row_max)[::-1]
    top = row_max_sorted[:min(n, row_max_sorted.size)]
    return float(np.mean(top))


# -------------------------
# 4) ProximityAvgAll (Durchschnitt aller Paare)
# -------------------------
def proximity_avg_all(set1: List[np.ndarray], set2: List[np.ndarray]) -> float:
    A = _as_matrix(set1)
    B = _as_matrix(set2)
    if A.size == 0 or B.size == 0:
        return 0.0
    D = A @ B.T
    normsA = np.linalg.norm(A, axis=1, keepdims=True)
    normsB = np.linalg.norm(B, axis=1, keepdims=True)
    denom = normsA @ normsB.T
    with np.errstate(divide='ignore', invalid='ignore'):
        S = np.divide(D, denom, out=np.zeros_like(D), where=denom != 0)
    return float(np.mean(S))

# -------------------------
# 5) SubsetProximityAvg (gemäß Tabelle: beste Subset-Kombinationen und ProximityMax)
# -------------------------
def subset_proximity_avg(set1: List[np.ndarray], set2: List[np.ndarray], subset_size: int = 3) -> float:
    """
    Korrekte Umsetzung gemäß Slides:
    1) Alle Subsets der Größe subset_size aus set1 bilden, für jedes Subset den Zentroid (Mittelwert) berechnen -> H_S1
    2) Analog für set2 -> H_S2
    3) ProximityMax(H_S1, H_S2) berechnen (maximale Kosinus-Ähnlichkeit zwischen allen Zentroid-Paaren)
    """
    A = _as_matrix(set1)  # (m, d)
    B = _as_matrix(set2)  # (n, d)

    m = A.shape[0] if A.size else 0
    n = B.shape[0] if B.size else 0

    if m < subset_size or n < subset_size:
        raise ValueError(f"Beide Mengen müssen mindestens subset_size={subset_size} Elemente enthalten "
                         f"(aktuell: {m}, {n}).")

    # --- 1) Alle Subset-Zentroiden für set1 erzeugen ---
    centroids1 = []
    for idxs in combinations(range(m), subset_size):
        sub = A[list(idxs), :]                 # shape: (subset_size, d)
        centroid = np.mean(sub, axis=0)        # Zentroid des Subsets
        centroids1.append(centroid)
    C1 = np.vstack(centroids1)                # shape: (k1, d)

    # --- 2) Alle Subset-Zentroiden für set2 erzeugen ---
    centroids2 = []
    for idxs in combinations(range(n), subset_size):
        sub = B[list(idxs), :]
        centroid = np.mean(sub, axis=0)
        centroids2.append(centroid)
    C2 = np.vstack(centroids2)                # shape: (k2, d)

    # --- 3) ProximityMax über die Zentroidmengen ---
    return proximity_max(C1, C2)
