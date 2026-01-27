from __future__ import annotations

import os
import time
from typing import List, Tuple, Iterable, Dict, Optional

import numpy as np
import pandas as pd

# Weighted similarity core + (optional) legacy extensions
from .cosine_funktions import (
    weighted_proximity_avg,        # lecture-consistent: featurewise weighted cosine
    weighted_proximity_max,        # extension (not lecture core)
    weighted_proximity_topn_avg,   # extension (not lecture core)
    weighted_subset_proximity_avg, # extension (not lecture core) - kept for compatibility
    weighted_proximity_avg_all     # extension/alias - kept for compatibility
)

# Heavy reuse from benchmark module (will be revised next step)
from .benchmark_of_semantic_operatiors import (
    _compute_all_customer_vectors,
    get_diverse_customers_with_clusters,
    save_or_load_phase_results,
    create_rankings_for_phase2,
    compute_overlap_coefficients,
    compute_phase3_metric_statistics,
    compute_phase4_separation_scores
)


# ---------------------------------------------------------------------
# Helpers (lecture-consistent weighting semantics)
# ---------------------------------------------------------------------

def _to_binary_churn(series: pd.Series) -> np.ndarray:
    """
    Convert churn column to numeric 0/1.
    Accepts:
      - already numeric/bool
      - strings like 'Churn_Yes' / 'Churn_No'
    """
    s = series.copy()
    if pd.api.types.is_bool_dtype(s):
        return s.astype(int).to_numpy()
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(int).to_numpy()

    # categorical / object
    s = s.astype(str)
    # strict default for your dataset naming
    return (s == "Churn_Yes").astype(int).to_numpy()


def _safe_weight_array(weight_vec: Iterable[float], n_features: int, weight_name: str) -> np.ndarray:
    """
    Validate weight vector as feature weights.
    Lecture semantics: len(w) == number of features (W).
    """
    w = np.asarray(list(weight_vec), dtype=float)
    if w.ndim != 1:
        raise ValueError(f"Gewichtungsvektor '{weight_name}' ist nicht 1-dimensional.")
    if len(w) != int(n_features):
        raise ValueError(
            f"Dimension-Mismatch für Gewichtung '{weight_name}': "
            f"len(w)={len(w)} vs. n_features={n_features}. "
            "In der Vorlesung sind Gewichte feature-basiert (Anzahl Attribute), nicht embedding_dim."
        )
    if np.any(w < 0):
        raise ValueError(f"Gewichtungsvektor '{weight_name}' enthält negative Werte (nicht erlaubt).")
    return w


def _zscore(col: pd.Series) -> pd.Series:
    v = col.dropna()
    if len(v) == 0:
        return pd.Series(np.nan, index=col.index)
    mu = float(v.mean())
    sd = float(v.std(ddof=0))
    if sd == 0.0 or np.isnan(sd):
        return pd.Series(np.nan, index=col.index)
    return (col - mu) / sd


# ---------------------------------------------------------------------
# Experiment 2 – Phase 1: weighted similarity computation (kept compatible)
# ---------------------------------------------------------------------

def compute_phase1_weighted_similarity_df(
    model,
    df_with_keys: pd.DataFrame,
    selected_customers: pd.DataFrame,
    weight_vectors: Dict[str, Iterable[float]],
    metrics_to_use: List[str] = ("proximity_avg",),
    topn_n: int = 3,
    subset_k: int = 3,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Phase 1 for Experiment 2 (Averaging vs. Weighting): compute weighted similarities.

    IMPORTANT (lecture alignment):
      - The lecture-consistent weighted similarity is 'proximity_avg' interpreted as:
            S_{c,w}(row_a,row_b) = sum_i w_i * cos(row_{a,i}, row_{b,i}) / sum_i w_i
        This is provided by weighted_proximity_avg(...) from cosine_funktions.py.

      - Other 'weighted_proximity_*' used here are kept ONLY for pipeline compatibility
        and are not part of the lecture core. They should be removed/disabled once
        benchmark_of_semantic_operatiors.py is consolidated.
    """
    if df_with_keys is None or len(df_with_keys) == 0:
        raise ValueError("df_with_keys ist leer oder None.")
    if selected_customers is None or len(selected_customers) == 0:
        raise ValueError("selected_customers ist leer oder None.")

    reference_ids = list(selected_customers.index)

    # Precompute all customer feature-vector lists once
    all_vectors = _compute_all_customer_vectors(model, df_with_keys)
    if not all_vectors or not all_vectors[0]:
        raise ValueError("Keine Feature-Vektoren gefunden (all_vectors leer).")

    n_features = len(all_vectors[0])

    # Validate weight vectors early (feature-weight semantics)
    weight_arrays: Dict[str, np.ndarray] = {}
    for weight_name, weight_vec in weight_vectors.items():
        weight_arrays[weight_name] = _safe_weight_array(weight_vec, n_features=n_features, weight_name=weight_name)

    # Map metric name -> function
    metric_functions = {
        "proximity_avg": lambda a, b, w: weighted_proximity_avg(a, b, w),  # lecture core
        # extensions (not lecture core) kept for compatibility:
        "proximity_topn_avg": lambda a, b, w: weighted_proximity_topn_avg(a, b, w, n=topn_n),
        "proximity_max": lambda a, b, w: weighted_proximity_max(a, b, w),
        "proximity_avg_all": lambda a, b, w: weighted_proximity_avg_all(a, b, w),
        "subset_proximity_avg": lambda a, b, w: weighted_subset_proximity_avg(a, b, w, subset_size=subset_k),
    }

    rows = []
    total_refs = len(reference_ids)
    total_customers = len(df_with_keys)

    for r_idx, ref_id in enumerate(reference_ids, start=1):
        if verbose:
            print(f"[Phase1] Ref {r_idx}/{total_refs}: {ref_id}")

        v_ref = all_vectors[ref_id]
        if not v_ref or len(v_ref) != n_features:
            raise ValueError(f"Referenzkunde {ref_id} hat keine gültigen Feature-Vektoren.")

        for other_id in range(total_customers):
            v_other = all_vectors[other_id]
            if not v_other or len(v_other) != n_features:
                continue

            row = {"Referenzkunde_ID": int(ref_id), "Kunde_ID": int(other_id)}

            for weight_name, w in weight_arrays.items():
                for metric_name in metrics_to_use:
                    if metric_name not in metric_functions:
                        continue
                    col_name = f"S_{metric_name.replace('proximity_', '')}_{weight_name}"

                    try:
                        sim = metric_functions[metric_name](v_ref, v_other, w)
                        row[col_name] = float(sim)
                    except Exception as e:
                        row[col_name] = np.nan
                        if verbose:
                            print(f"  Warnung: {metric_name}/{weight_name} für {ref_id}-{other_id} fehlgeschlagen: {e}")

            rows.append(row)

    df = pd.DataFrame(rows)

    id_cols = ["Referenzkunde_ID", "Kunde_ID"]
    metric_cols = sorted([c for c in df.columns if c not in id_cols])
    return df[id_cols + metric_cols]


# ---------------------------------------------------------------------
# Experiment 2 – Phase 3B: churn propensity (lecture core)
# ---------------------------------------------------------------------

def compute_churn_propensity_scores_featurewise(
    model,
    df_with_keys: pd.DataFrame,
    weight_vectors: Dict[str, Iterable[float]],
    churn_col: str = "churn",
    include_churned: bool = False,
    zscore: bool = True,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Lecture-consistent churn propensity:

      Score(customer) = sum_i w_i * cos(customer_i, churn_centroid_i) / sum_i w_i

    where churn_centroid_i is the centroid over *churned customers* for feature i.

    Returns columns:
      Kunde_ID, churn, ChurnProp_<w>, (optional) ChurnPropZ_<w>, Rank_<w>
    """
    if churn_col not in df_with_keys.columns:
        raise ValueError(f"Spalte '{churn_col}' fehlt in df_with_keys.")

    df_proc = df_with_keys.copy()
    churn_bin = _to_binary_churn(df_proc[churn_col])
    df_proc[churn_col] = churn_bin

    churn_mask = (churn_bin == 1)
    active_mask = ~churn_mask
    if churn_mask.sum() == 0:
        raise ValueError("Keine churned Kunden gefunden (churn=1).")

    # all_vectors: list[cust] -> list[feature] -> vector
    all_vectors = _compute_all_customer_vectors(model, df_proc)
    if not all_vectors or not all_vectors[0]:
        raise ValueError("Keine Feature-Vektoren gefunden.")
    n_features = len(all_vectors[0])

    # prepare output
    out = pd.DataFrame({
        "Kunde_ID": np.arange(len(df_proc), dtype=int),
        churn_col: churn_bin
    })

    # precompute per-feature churn centroids
    churn_feature_centroids: List[np.ndarray] = []
    for fi in range(n_features):
        feat_vecs = []
        for cust_idx in np.where(churn_mask)[0]:
            vec_list = all_vectors[cust_idx]
            if vec_list and len(vec_list) == n_features:
                feat_vecs.append(vec_list[fi])
        if len(feat_vecs) == 0:
            # fallback: zero vector with correct dim inferred from first available
            example = next((v[fi] for v in all_vectors if v and len(v) == n_features), None)
            if example is None:
                raise ValueError("Kann Dimension der Feature-Vektoren nicht bestimmen.")
            churn_feature_centroids.append(np.zeros_like(example))
        else:
            churn_feature_centroids.append(np.mean(np.stack(feat_vecs), axis=0))

    # compute scores per weight vector
    for weight_name, weight_vec in weight_vectors.items():
        w = _safe_weight_array(weight_vec, n_features=n_features, weight_name=weight_name)

        scores = np.full(len(df_proc), np.nan, dtype=float)

        # for each customer: featurewise weighted cosine
        for cust_idx in range(len(df_proc)):
            vec_list = all_vectors[cust_idx]
            if not vec_list or len(vec_list) != n_features:
                continue

            # Use the lecture-core implementation in cosine_funktions.py
            score = weighted_proximity_avg(vec_list, churn_feature_centroids, w)
            scores[cust_idx] = score

        if not include_churned:
            scores = np.where(active_mask, scores, np.nan)

        score_col = f"ChurnProp_{weight_name}"
        out[score_col] = scores

        rank_col = f"Rank_{weight_name}"
        out[rank_col] = out[score_col].rank(ascending=False, method="min")

        if zscore:
            z_col = f"ChurnPropZ_{weight_name}"
            out[z_col] = _zscore(out[score_col])

        if verbose:
            v = out[score_col].dropna()
            if len(v) > 0:
                print(f"[ChurnFeaturewise] {weight_name}: "
                      f"min={float(v.min()):.4f} mean={float(v.mean()):.4f} "
                      f"median={float(v.median()):.4f} max={float(v.max()):.4f}")

    return out


# ---------------------------------------------------------------------
# Deprecated churn functions (kept for compatibility, routed to lecture core)
# ---------------------------------------------------------------------

def compute_churn_centroids_weighted(*args, **kwargs):
    """
    DEPRECATED.
    The previous implementation applied weights in embedding-dimension space (centroid * w),
    which is not lecture-consistent for feature weights.

    Keep only for compatibility: call lecture-consistent featurewise scoring pipeline instead.
    """
    raise NotImplementedError(
        "compute_churn_centroids_weighted ist deprecated und wurde entfernt, "
        "da die alte Implementierung nicht vorlesungskonform war. "
        "Nutzen Sie compute_churn_propensity_scores_featurewise(...)."
    )


def compute_churn_propensity_scores(*args, **kwargs) -> pd.DataFrame:
    """
    DEPRECATED wrapper kept for older code paths.
    Routes to lecture-consistent compute_churn_propensity_scores_featurewise(...).
    """
    return compute_churn_propensity_scores_featurewise(*args, **kwargs)


# ---------------------------------------------------------------------
# Experiment 2 – orchestration (kept compatible with benchmark module)
# ---------------------------------------------------------------------

def run_experiment2_weighting_analysis(
    model_list,
    df_with_keys: pd.DataFrame,
    weight_vectors: dict,
    metrics_to_use: List[str] = ("proximity_avg", "proximity_topn_avg"),
    topn_n: int = 3,
    subset_k: int = 3,
    top_n: int = 200,
    base_dir: str = ".",
    churn_col: str = "churn",
    run_churn_propensity: bool = True,
    churn_include_churned: bool = False,
    churn_use_zscores: bool = True,
    verbose: bool = False
) -> dict:
    """
    Experiment 2: weighting analysis.
    This function heavily reuses benchmark_of_semantic_operatiors.py utilities.
    """
    results = {}

    selected_customers, cluster_centers, selected_cluster_info, kmeans = \
        get_diverse_customers_with_clusters(df_with_keys, k=5, verbose=verbose)

    if verbose:
        print("\n" + "=" * 70)
        print("EXPERIMENT 2: AVERAGING vs. WEIGHTING ANALYSE")
        print("=" * 70)
        print(f"\nGewichtungsvektoren: {list(weight_vectors.keys())}")
        print(f"Metriken: {list(metrics_to_use)}")

    for model_idx, (model, model_info) in enumerate(model_list):
        if verbose:
            print(f"\n{'='*60}\nMODEL {model_idx}: {model_info}\n{'='*60}\n")

        # PHASE 1
        phase1_data = save_or_load_phase_results(
            phase_name="exp2_phase1_weighted_similarity",
            phase_data=None,
            base_dir=base_dir,
            model_idx=model_idx,
            model_info=model_info,
            selected_customers=selected_customers,
            verbose=verbose
        )

        if phase1_data is None:
            phase1_df = compute_phase1_weighted_similarity_df(
                model=model,
                df_with_keys=df_with_keys,
                selected_customers=selected_customers,
                weight_vectors=weight_vectors,
                metrics_to_use=list(metrics_to_use),
                topn_n=topn_n,
                subset_k=subset_k,
                verbose=verbose
            )
            phase1_df = save_or_load_phase_results(
                phase_name="exp2_phase1_weighted_similarity",
                phase_data=phase1_df,
                base_dir=base_dir,
                model_idx=model_idx,
                model_info=model_info,
                selected_customers=selected_customers,
                verbose=verbose
            )
        else:
            phase1_df = phase1_data

        results[f"exp2_phase1_model{model_idx}"] = phase1_df

        # PHASE 2: Rankings (reuse)
        phase2_rank_data = save_or_load_phase_results(
            phase_name="exp2_phase2_rankings",
            phase_data=None,
            base_dir=base_dir,
            model_idx=model_idx,
            model_info=model_info,
            selected_customers=selected_customers,
            verbose=verbose
        )
        if phase2_rank_data is None:
            rankings = create_rankings_for_phase2(phase1_df=phase1_df, top_n=top_n)
            rankings = save_or_load_phase_results(
                phase_name="exp2_phase2_rankings",
                phase_data=rankings,
                base_dir=base_dir,
                model_idx=model_idx,
                model_info=model_info,
                selected_customers=selected_customers,
                verbose=verbose
            )
        else:
            rankings = phase2_rank_data
        results[f"exp2_phase2_rankings_model{model_idx}"] = rankings

        # PHASE 2 overlaps
        overlaps_data = save_or_load_phase_results(
            phase_name="exp2_phase2_overlaps",
            phase_data=None,
            base_dir=base_dir,
            model_idx=model_idx,
            model_info=model_info,
            selected_customers=selected_customers,
            verbose=verbose
        )
        if overlaps_data is None:
            overlaps = compute_overlap_coefficients(rankings_dict=rankings, top_n=top_n)
            overlaps = save_or_load_phase_results(
                phase_name="exp2_phase2_overlaps",
                phase_data=overlaps,
                base_dir=base_dir,
                model_idx=model_idx,
                model_info=model_info,
                selected_customers=selected_customers,
                verbose=verbose
            )
        else:
            overlaps = overlaps_data
        results[f"exp2_phase2_overlaps_model{model_idx}"] = overlaps

        # PHASE 3 stats
        phase3_data = save_or_load_phase_results(
            phase_name="exp2_phase3_stats",
            phase_data=None,
            base_dir=base_dir,
            model_idx=model_idx,
            model_info=model_info,
            selected_customers=selected_customers,
            verbose=verbose
        )
        if phase3_data is None:
            phase3 = compute_phase3_metric_statistics(
                phase1_df=phase1_df,
                bins=50,
                use_global_bins_across_metrics=True
            )
            phase3 = save_or_load_phase_results(
                phase_name="exp2_phase3_stats",
                phase_data=phase3,
                base_dir=base_dir,
                model_idx=model_idx,
                model_info=model_info,
                selected_customers=selected_customers,
                verbose=verbose
            )
        else:
            phase3 = phase3_data
        results[f"exp2_phase3_stats_model{model_idx}"] = phase3

        # PHASE 4 separation
        phase4_data = save_or_load_phase_results(
            phase_name="exp2_phase4_separation",
            phase_data=None,
            base_dir=base_dir,
            model_idx=model_idx,
            model_info=model_info,
            selected_customers=selected_customers,
            verbose=verbose
        )
        if phase4_data is None:
            phase4 = compute_phase4_separation_scores(
                phase1_df=phase1_df,
                rankings_dict=rankings,
                top_n=top_n
            )
            phase4 = save_or_load_phase_results(
                phase_name="exp2_phase4_separation",
                phase_data=phase4,
                base_dir=base_dir,
                model_idx=model_idx,
                model_info=model_info,
                selected_customers=selected_customers,
                verbose=verbose
            )
        else:
            phase4 = phase4_data
        results[f"exp2_phase4_separation_model{model_idx}"] = phase4

        # Optional: churn propensity (lecture core)
        if run_churn_propensity:
            churn_phase_name = "exp2_phase3b_churn_propensity_featurewise"

            churn_data = save_or_load_phase_results(
                phase_name=churn_phase_name,
                phase_data=None,
                base_dir=base_dir,
                model_idx=model_idx,
                model_info=model_info,
                selected_customers=selected_customers,
                verbose=verbose
            )
            if churn_data is None:
                churn_scores_df = compute_churn_propensity_scores_featurewise(
                    model=model,
                    df_with_keys=df_with_keys,
                    weight_vectors=weight_vectors,
                    churn_col=churn_col,
                    include_churned=churn_include_churned,
                    zscore=churn_use_zscores,
                    verbose=verbose
                )
                churn_scores_df = save_or_load_phase_results(
                    phase_name=churn_phase_name,
                    phase_data=churn_scores_df,
                    base_dir=base_dir,
                    model_idx=model_idx,
                    model_info=model_info,
                    selected_customers=selected_customers,
                    verbose=verbose
                )
            else:
                churn_scores_df = churn_data

            results[f"exp2_churn_propensity_model{model_idx}"] = churn_scores_df

    return results
