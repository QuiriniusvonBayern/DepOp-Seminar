from __future__ import annotations

import os
import time
from typing import List, Tuple, Iterable, Dict, Optional

import numpy as np
import pandas as pd

# Weighted similarity core + (optional) legacy extensions
from .cosine_funktions import (
    weighted_proximity_avg,        # lecture-consistent: featurewise weighted cosine
)

# Heavy reuse from benchmark module (will be revised next step)
from .benchmark_of_semantic_operatiors import (
    _compute_all_customer_vectors,
    get_diverse_customers_with_clusters,
    save_or_load_phase_results,
    compute_phase3_metric_statistics
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
# Helper functions for tidy DataFrames (added as requested)
# ---------------------------------------------------------------------

def rankings_dict_to_long_df(rankings_dict: dict) -> pd.DataFrame:
    """
    Convert rankings dictionary to long DataFrame format.
    """
    rows = []
    for ref_id, pack in rankings_dict.items():
        for side in ["top", "bottom"]:
            for metric, df in pack[side].items():
                tmp = df.copy()
                tmp["Referenzkunde_ID"] = int(ref_id)
                tmp["side"] = side
                tmp["metric"] = metric
                rows.append(tmp[["Referenzkunde_ID", "side", "metric", "Kunde_ID", "score", "rank"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["Referenzkunde_ID", "side", "metric", "Kunde_ID", "score", "rank"]
    )


def overlap_curves_dict_to_long_df(overlaps_dict: dict) -> pd.DataFrame:
    """
    Convert overlap curves dictionary to long DataFrame format.
    """
    rows = []
    for ref_id, pack in overlaps_dict.items():
        for side_key in ["top_curves", "bottom_curves"]:
            side = "top" if side_key == "top_curves" else "bottom"
            df = pack[side_key].copy()
            df["K"] = df.index.astype(int)
            long = df.melt(id_vars=["K"], var_name="metric", value_name="overlap")
            long["Referenzkunde_ID"] = int(ref_id)
            long["side"] = side
            rows.append(long[["Referenzkunde_ID", "side", "metric", "K", "overlap"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["Referenzkunde_ID", "side", "metric", "K", "overlap"]
    )


# ---------------------------------------------------------------------
# Experiment 2 – Phase 1: weighted similarity computation (kept compatible)
# ---------------------------------------------------------------------

def compute_phase1_weighted_similarity_df(
    model,
    df_with_keys: pd.DataFrame,
    selected_customers: pd.DataFrame,
    weight_vectors: Dict[str, Iterable[float]],
    metrics_to_use: List[str] = ("proximity_avg",),
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

            if other_id == ref_id: 
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
# Experiment 2 – Phase 2: Rankings and Overlap Curves (updated)
# ---------------------------------------------------------------------

def create_rankings_for_phase2_with_scores(
    phase1_df: pd.DataFrame,
    top_n: int = 200,
    *,
    ref_col: str = "Referenzkunde_ID",
    id_col: str = "Kunde_ID",
    exclude_self_match: bool = True,
    verbose: bool = False
) -> dict:
    """
    Erstellt Top-N und Bottom-N Rankings pro Metrik pro Referenzkunde.
    Liefert DataFrames mit Kunde_ID, score, rank (statt nur ID-Listen).
    """
    metric_cols = [c for c in phase1_df.columns if c not in [ref_col, id_col]]

    reference_ids = phase1_df[ref_col].unique()
    rankings_dict = {}

    for ref_id in reference_ids:
        if verbose:
            print(f"Erstelle Rankings für Referenzkunde {ref_id}")

        ref_data = phase1_df[phase1_df[ref_col] == ref_id].copy()

        if exclude_self_match:
            ref_data = ref_data[ref_data[id_col] != ref_id]

        ref_dict = {"top": {}, "bottom": {}}

        for metric in metric_cols:
            # TOP
            top_sorted = ref_data.sort_values(by=metric, ascending=False).head(top_n)
            top_df = top_sorted[[id_col, metric]].rename(columns={metric: "score"}).copy()
            top_df["rank"] = np.arange(1, len(top_df) + 1, dtype=int)
            ref_dict["top"][metric] = top_df.rename(columns={id_col: "Kunde_ID"})[["Kunde_ID", "score", "rank"]]

            # BOTTOM
            bottom_sorted = ref_data.sort_values(by=metric, ascending=True).head(top_n)
            bottom_df = bottom_sorted[[id_col, metric]].rename(columns={metric: "score"}).copy()
            bottom_df["rank"] = np.arange(1, len(bottom_df) + 1, dtype=int)
            ref_dict["bottom"][metric] = bottom_df.rename(columns={id_col: "Kunde_ID"})[["Kunde_ID", "score", "rank"]]

        rankings_dict[int(ref_id)] = ref_dict

    return rankings_dict


def compute_overlap_curves_against_baseline(
    rankings_dict: dict,
    *,
    baseline_metric: str = "S_avg_average",
    top_n: int = 200,
    use_fraction: bool = False,
    verbose: bool = False
) -> dict:
    """
    Berechnet Overlap-Kurven (K=1..top_n) gegen eine Baseline-Metrik.
    Output pro Referenzkunde:
      - top_curves: DataFrame(index=K, columns=andere Metriken) mit Overlap-Anzahl (oder Quote)
      - bottom_curves: analog
      - summary_stats: einfache Zusammenfassung
    """
    out = {}

    for ref_id, pack in rankings_dict.items():
        if verbose:
            print(f"Berechne Overlap-Kurven für Referenzkunde {ref_id}")

        metrics = list(pack["top"].keys())
        if baseline_metric not in metrics:
            raise ValueError(f"Baseline '{baseline_metric}' fehlt für ref_id={ref_id}. metrics={metrics}")

        others = [m for m in metrics if m != baseline_metric]

        base_top = pack["top"][baseline_metric]["Kunde_ID"].astype(int).to_numpy()[:top_n]
        base_bot = pack["bottom"][baseline_metric]["Kunde_ID"].astype(int).to_numpy()[:top_n]

        top_curves = {}
        bot_curves = {}

        for m in others:
            ids_top = pack["top"][m]["Kunde_ID"].astype(int).to_numpy()[:top_n]
            ids_bot = pack["bottom"][m]["Kunde_ID"].astype(int).to_numpy()[:top_n]

            top_vals = np.empty(len(ids_top), dtype=float if use_fraction else int)
            bot_vals = np.empty(len(ids_bot), dtype=float if use_fraction else int)

            sb, so = set(), set()
            for i in range(len(ids_top)):
                sb.add(int(base_top[i]))
                so.add(int(ids_top[i]))
                ov = len(sb.intersection(so))
                top_vals[i] = ov / (i+1) if use_fraction else ov

            sb, so = set(), set()
            for i in range(len(ids_bot)):
                sb.add(int(base_bot[i]))
                so.add(int(ids_bot[i]))
                ov = len(sb.intersection(so))
                bot_vals[i] = ov / (i+1) if use_fraction else ov

            top_curves[m] = top_vals
            bot_curves[m] = bot_vals

        top_df = pd.DataFrame(top_curves, index=np.arange(1, len(base_top) + 1, dtype=int))
        bot_df = pd.DataFrame(bot_curves, index=np.arange(1, len(base_bot) + 1, dtype=int))
        top_df.index.name = "K"
        bot_df.index.name = "K"

        out[int(ref_id)] = {
            "top_curves": top_df,
            "bottom_curves": bot_df,
            "summary_stats": {
                "baseline_metric": baseline_metric,
                "top_n": int(top_n),
                "use_fraction": bool(use_fraction),
                "n_metrics_compared": len(others),
            }
        }

    return out

# ---------------------------------------------------------------------
# Experiment 2 – Phase 4: 
# ---------------------------------------------------------------------

def compute_phase4_separation_scores(
    phase1_df: pd.DataFrame,
    rankings_dict: dict,
    top_n: int = 200
) -> dict:
    """
    Phase 4: Diskriminierungsfähigkeit (Separation Score).

    Separation = mean(similarity(top)) - mean(similarity(bottom))

    Unterstützt Rankings im alten Format (Listen von IDs) und im neuen Format
    (DataFrames mit Kunde_ID, score, rank).
    """
    if phase1_df is None or len(phase1_df) == 0:
        raise ValueError("phase1_df ist leer oder None.")
    if rankings_dict is None or len(rankings_dict) == 0:
        raise ValueError("rankings_dict ist leer oder None.")

    metric_cols = [c for c in phase1_df.columns if c not in ["Referenzkunde_ID", "Kunde_ID"]]
    if not metric_cols:
        raise ValueError("Keine Metrikspalten gefunden (erwartet: alles außer Referenzkunde_ID, Kunde_ID).")

    # Lookup je Referenzkunde
    phase1_grouped = {}
    for ref_id in phase1_df["Referenzkunde_ID"].unique():
        sub = phase1_df[phase1_df["Referenzkunde_ID"] == ref_id].copy()
        phase1_grouped[int(ref_id)] = sub.set_index("Kunde_ID")

    def _extract_ids(x, n: int) -> list:
        """
        x kann sein:
          - list/tuple/np.ndarray von IDs
          - pd.DataFrame mit Spalte Kunde_ID
          - pd.Series
        """
        if x is None:
            return []
        if isinstance(x, pd.DataFrame):
            if "Kunde_ID" not in x.columns:
                # Falls Kunde_ID im Index steckt
                if x.index.name == "Kunde_ID":
                    return [int(v) for v in x.index.to_list()[:n]]
                raise ValueError(f"Ranking-DF ohne Kunde_ID-Spalte. columns={x.columns.tolist()}")
            return [int(v) for v in x["Kunde_ID"].to_list()[:n]]
        if isinstance(x, pd.Series):
            return [int(v) for v in x.to_list()[:n]]
        if isinstance(x, (list, tuple, np.ndarray)):
            return [int(v) for v in list(x)[:n]]
        raise TypeError(f"Unbekannter Ranking-Typ: {type(x)}")

    rows = []

    for ref_id, rdata in rankings_dict.items():
        ref_id_int = int(ref_id)
        if ref_id_int not in phase1_grouped:
            continue

        sub = phase1_grouped[ref_id_int]
        top_dict = rdata.get("top", {})
        bottom_dict = rdata.get("bottom", {})

        metrics = list(top_dict.keys())

        for m in metrics:
            if m not in metric_cols:
                continue

            top_ids = _extract_ids(top_dict.get(m), top_n)
            bottom_ids = _extract_ids(bottom_dict.get(m), top_n)

            top_vals = pd.to_numeric(sub.reindex(top_ids)[m], errors="coerce").dropna()
            bottom_vals = pd.to_numeric(sub.reindex(bottom_ids)[m], errors="coerce").dropna()

            mean_top = float(top_vals.mean()) if len(top_vals) > 0 else np.nan
            mean_bottom = float(bottom_vals.mean()) if len(bottom_vals) > 0 else np.nan
            separation = mean_top - mean_bottom if (np.isfinite(mean_top) and np.isfinite(mean_bottom)) else np.nan

            rows.append({
                "Referenzkunde_ID": ref_id_int,
                "Metrik": m,
                "mean_top": mean_top,
                "mean_bottom": mean_bottom,
                "separation": separation,
                "n_top_used": int(len(top_vals)),
                "n_bottom_used": int(len(bottom_vals)),
            })

    per_reference_df = pd.DataFrame(rows)
    if len(per_reference_df) == 0:
        raise ValueError("Keine Separation Scores berechnet (prüfen Sie Rankings/IDs/Metriknamen).")

    summary_df = (
        per_reference_df.groupby("Metrik")["separation"]
        .agg(separation_mean="mean", separation_median="median", separation_std="std")
        .reset_index()
        .sort_values("Metrik")
    )

    return {
        "per_reference_df": per_reference_df,
        "summary_df": summary_df,
    }

# ---------------------------------------------------------------------
# Deprecated functions (kept for compatibility)
# ---------------------------------------------------------------------

def compute_overlap_curves_exp2(rankings_dict: dict, baseline_col: str = "S_avg_average", top_n: int = 200) -> dict:
    """
    DEPRECATED: Use compute_overlap_curves_against_baseline instead.
    Kept for compatibility.
    """
    print("Warnung: compute_overlap_curves_exp2 ist deprecated. Verwende compute_overlap_curves_against_baseline.")
    return compute_overlap_curves_against_baseline(
        rankings_dict=rankings_dict,
        baseline_metric=baseline_col,
        top_n=top_n,
        use_fraction=False,
        verbose=False
    )


def create_rankings_for_phase2_exp2(phase1_df: pd.DataFrame, top_n: int = 200) -> dict:
    """
    DEPRECATED: Use create_rankings_for_phase2_with_scores instead.
    Kept for compatibility.
    """
    print("Warnung: create_rankings_for_phase2_exp2 ist deprecated. Verwende create_rankings_for_phase2_with_scores.")
    return create_rankings_for_phase2_with_scores(phase1_df=phase1_df, top_n=top_n)


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
    metrics_to_use: List[str] = ["proximity_avg"],
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

        # PHASE 2: Rankings (with updated phase names for cache versioning)
        phase2_rank_data = save_or_load_phase_results(
            phase_name="exp2_phase2_rankings_v2_scores",
            phase_data=None,
            base_dir=base_dir,
            model_idx=model_idx,
            model_info=model_info,
            selected_customers=selected_customers,
            verbose=verbose
        )
        if phase2_rank_data is None:
            rankings = create_rankings_for_phase2_with_scores(phase1_df=phase1_df, top_n=top_n, verbose=verbose)
            rankings = save_or_load_phase_results(
                phase_name="exp2_phase2_rankings_v2_scores",
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
        
        # Add tidy long DataFrame for rankings
        results[f"exp2_phase2_rankings_long_model{model_idx}"] = rankings_dict_to_long_df(rankings)

        # PHASE 2 overlaps (with updated phase names for cache versioning)
        overlaps_data = save_or_load_phase_results(
            phase_name="exp2_phase2_overlaps_v2_curves",
            phase_data=None,
            base_dir=base_dir,
            model_idx=model_idx,
            model_info=model_info,
            selected_customers=selected_customers,
            verbose=verbose
        )
        if overlaps_data is None:
            overlaps = compute_overlap_curves_against_baseline(
                rankings_dict=rankings,
                baseline_metric="S_avg_average",
                top_n=top_n,
                use_fraction=False,
                verbose=verbose
            )
            overlaps = save_or_load_phase_results(
                phase_name="exp2_phase2_overlaps_v2_curves",
                phase_data=overlaps,
                base_dir=base_dir,
                model_idx=model_idx,
                model_info=model_info,
                selected_customers=selected_customers,
                verbose=verbose
            )
            overlap_long = overlap_curves_dict_to_long_df(overlaps)
            overlap_agg = aggregate_overlap_long_df(overlap_long)
            phase3_summary_long = compute_phase3_summary_stats_long(phase1_df)
            phase3_hist_long = compute_phase3_hist_long(
                phase1_df=phase1_df,
                bins=50,
                use_global_bins_across_metrics=True
            )

        else:
            overlaps = overlaps_data
        results[f"exp2_phase2_overlaps_model{model_idx}"] = overlaps
        results[f"exp2_phase2_overlaps_long_model{model_idx}"] = overlap_long
        results[f"exp2_phase2_overlaps_agg_model{model_idx}"] = overlap_agg
        results[f"exp2_phase3_summary_long_model{model_idx}"] = phase3_summary_long
        results[f"exp2_phase3_hist_long_model{model_idx}"] = phase3_hist_long
        
        # Add tidy long DataFrame for overlaps
        results[f"exp2_phase2_overlaps_long_model{model_idx}"] = overlap_curves_dict_to_long_df(overlaps)

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


def overlap_curves_dict_to_long_df(overlaps_dict: dict) -> pd.DataFrame:
    """
    overlaps_dict[ref_id] = {
      'top_curves': DataFrame(index=K, columns=metrics),
      'bottom_curves': DataFrame(index=K, columns=metrics),
      ...
    }
    -> long DF: Referenzkunde_ID, side, K, metric, overlap
    """
    parts = []
    for ref_id, pack in overlaps_dict.items():
        ref_id = int(ref_id)

        for side_key, side_name in [("top_curves", "top"), ("bottom_curves", "bottom")]:
            if side_key not in pack or pack[side_key] is None:
                continue
            df = pack[side_key].copy()
            df = df.copy()
            df["K"] = df.index.astype(int)

            long = df.melt(id_vars=["K"], var_name="metric", value_name="overlap")
            long["Referenzkunde_ID"] = ref_id
            long["side"] = side_name
            parts.append(long[["Referenzkunde_ID", "side", "K", "metric", "overlap"]])

    if not parts:
        return pd.DataFrame(columns=["Referenzkunde_ID", "side", "K", "metric", "overlap"])

    out = pd.concat(parts, ignore_index=True)
    out.sort_values(["side", "metric", "K", "Referenzkunde_ID"], inplace=True)
    return out

def aggregate_overlap_long_df(overlap_long_df: pd.DataFrame) -> pd.DataFrame:
    if overlap_long_df is None or len(overlap_long_df) == 0:
        return pd.DataFrame(columns=[
            "side","K","metric","overlap_mean","overlap_median","overlap_std","n_refs"
        ])

    agg = (overlap_long_df
           .groupby(["side", "K", "metric"], as_index=False)
           .agg(overlap_mean=("overlap", "mean"),
                overlap_median=("overlap", "median"),
                overlap_std=("overlap", "std"),
                n_refs=("Referenzkunde_ID", "nunique")))
    agg.sort_values(["side", "metric", "K"], inplace=True)
    return agg


def compute_phase3_summary_stats_long(
    phase1_df: pd.DataFrame,
    ref_col: str = "Referenzkunde_ID",
    id_col: str = "Kunde_ID"
) -> pd.DataFrame:
    """
    Plot-ready Summary Stats:
    ref_id, metric, min, max, mean, std, neg_frac, span
    """
    metric_cols = [c for c in phase1_df.columns if c not in [ref_col, id_col]]
    parts = []

    for m in metric_cols:
        tmp = phase1_df[[ref_col, m]].copy()
        tmp[m] = pd.to_numeric(tmp[m], errors="coerce")
        g = tmp.groupby(ref_col)[m]

        stat = g.agg(["min", "max", "mean", "std"]).reset_index()
        stat.rename(columns={ref_col: "Referenzkunde_ID"}, inplace=True)
        stat["metric"] = m

        # negativer Anteil
        neg = tmp.assign(is_neg=lambda x: (x[m] < 0).astype(int)).groupby(ref_col)["is_neg"].mean().reset_index()
        neg.rename(columns={ref_col: "Referenzkunde_ID", "is_neg": "neg_frac"}, inplace=True)

        stat = stat.merge(neg, on="Referenzkunde_ID", how="left")
        stat["span"] = stat["max"] - stat["min"]

        parts.append(stat[["Referenzkunde_ID","metric","min","max","mean","std","neg_frac","span"]])

    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["Referenzkunde_ID","metric","min","max","mean","std","neg_frac","span"]
    )
    out.sort_values(["metric","Referenzkunde_ID"], inplace=True)
    return out

def compute_phase3_hist_long(
    phase1_df: pd.DataFrame,
    bins: int = 50,
    use_global_bins_across_metrics: bool = True,
    ref_col: str = "Referenzkunde_ID",
    id_col: str = "Kunde_ID"
) -> pd.DataFrame:
    """
    Plot-ready Histogramme:
    metric, bin_left, bin_right, bin_center, count, density
    Optional könnte man auch je Referenzkunde ausgeben, aber global ist meist ausreichend.
    """
    metric_cols = [c for c in phase1_df.columns if c not in [ref_col, id_col]]

    # globales Min/Max für gemeinsame Bins
    if use_global_bins_across_metrics:
        all_vals = pd.concat([pd.to_numeric(phase1_df[m], errors="coerce") for m in metric_cols], ignore_index=True)
        all_vals = all_vals.dropna()
        if len(all_vals) == 0:
            return pd.DataFrame(columns=["metric","bin_left","bin_right","bin_center","count","density"])
        global_min, global_max = float(all_vals.min()), float(all_vals.max())
        edges = np.linspace(global_min, global_max, bins + 1)
    else:
        edges = None

    rows = []
    for m in metric_cols:
        vals = pd.to_numeric(phase1_df[m], errors="coerce").dropna().to_numpy()
        if len(vals) == 0:
            continue

        if edges is None:
            local_min, local_max = float(np.min(vals)), float(np.max(vals))
            edges_m = np.linspace(local_min, local_max, bins + 1)
        else:
            edges_m = edges

        counts, bin_edges = np.histogram(vals, bins=edges_m)
        widths = np.diff(bin_edges)
        density = counts / (counts.sum() * widths)  # Dichte

        for i in range(len(counts)):
            rows.append({
                "metric": m,
                "bin_left": float(bin_edges[i]),
                "bin_right": float(bin_edges[i+1]),
                "bin_center": float((bin_edges[i] + bin_edges[i+1]) / 2),
                "count": int(counts[i]),
                "density": float(density[i]) if counts.sum() > 0 else np.nan
            })

    out = pd.DataFrame(rows)
    out.sort_values(["metric","bin_center"], inplace=True)
    return out
