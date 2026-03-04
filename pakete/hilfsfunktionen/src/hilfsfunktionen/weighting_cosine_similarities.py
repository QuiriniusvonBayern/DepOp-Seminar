from __future__ import annotations

import os
import time
from typing import List, Tuple, Iterable, Dict, Optional

import numpy as np
import pandas as pd

from .cosine_funktions import (
    weighted_proximity_avg,
)

from .benchmark_of_semantic_operatiors import (
    _compute_all_customer_vectors,
    get_diverse_customers_with_clusters,
    save_or_load_phase_results,
    compute_phase3_metric_statistics,
)


def _convert_to_binary_churn(series: pd.Series) -> np.ndarray:
    """
    Convert a churn indicator column to a binary integer numpy array.

    Handles boolean, numeric, and string representations where 'Churn_Yes'
    indicates churned customers.

    Args:
        series: Input series containing churn information.

    Returns:
        Numpy array with 1 for churned and 0 for non-churned customers.
    """
    s = series.copy()
    if pd.api.types.is_bool_dtype(s):
        return s.astype(int).to_numpy()
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(int).to_numpy()

    s = s.astype(str)
    return (s == "Churn_Yes").astype(int).to_numpy()


def _validate_weight_vector(
    weight_values: Iterable[float], num_features: int, weight_name: str
) -> np.ndarray:
    """
    Validate and convert a weight vector for feature-wise weighting.

    Ensures the vector is one-dimensional, matches the number of features,
    and contains no negative values.

    Args:
        weight_values: Input weight values.
        num_features: Expected number of features.
        weight_name: Identifier for error messages.

    Returns:
        Validated numpy array of weights.

    Raises:
        ValueError: If validation fails.
    """
    w = np.asarray(list(weight_values), dtype=float)
    if w.ndim != 1:
        raise ValueError(f"Weight vector '{weight_name}' is not one-dimensional.")
    if len(w) != int(num_features):
        raise ValueError(
            f"Dimension mismatch for weight vector '{weight_name}': "
            f"len(w)={len(w)} vs. num_features={num_features}."
        )
    if np.any(w < 0):
        raise ValueError(f"Weight vector '{weight_name}' contains negative values.")
    return w


def _zscore_normalize(col: pd.Series) -> pd.Series:
    """
    Apply z-score normalization to a pandas Series.

    Args:
        col: Input series.

    Returns:
        Series with z-scores, or NaNs if normalization is not possible.
    """
    v = col.dropna()
    if len(v) == 0:
        return pd.Series(np.nan, index=col.index)
    mean_val = float(v.mean())
    std_val = float(v.std(ddof=0))
    if std_val == 0.0 or np.isnan(std_val):
        return pd.Series(np.nan, index=col.index)
    return (col - mean_val) / std_val


def rankings_to_long_dataframe(rankings_dict: dict) -> pd.DataFrame:
    """
    Convert a nested rankings dictionary to a long-format DataFrame.

    Args:
        rankings_dict: Dictionary with structure
            {ref_id: {"top": {metric: df}, "bottom": {metric: df}}}.

    Returns:
        DataFrame with columns Referenzkunde_ID, side, metric, Kunde_ID, score, rank.
    """
    rows = []
    for ref_id, pack in rankings_dict.items():
        for side in ["top", "bottom"]:
            for metric, df in pack[side].items():
                tmp = df.copy()
                tmp["Referenzkunde_ID"] = int(ref_id)
                tmp["side"] = side
                tmp["metric"] = metric
                rows.append(
                    tmp[
                        [
                            "Referenzkunde_ID",
                            "side",
                            "metric",
                            "Kunde_ID",
                            "score",
                            "rank",
                        ]
                    ]
                )
    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame(
        columns=["Referenzkunde_ID", "side", "metric", "Kunde_ID", "score", "rank"]
    )


def overlap_curves_to_long_dataframe(overlaps_dict: dict) -> pd.DataFrame:
    """
    Convert overlap curves dictionary to a long-format DataFrame.

    Args:
        overlaps_dict: Dictionary with structure
            {ref_id: {"top_curves": df, "bottom_curves": df}}.

    Returns:
        DataFrame with columns Referenzkunde_ID, side, metric, K, overlap.
    """
    rows = []
    for ref_id, pack in overlaps_dict.items():
        for side_key in ["top_curves", "bottom_curves"]:
            side = "top" if side_key == "top_curves" else "bottom"
            df = pack[side_key].copy()
            df["K"] = df.index.astype(int)
            long_df = df.melt(id_vars=["K"], var_name="metric", value_name="overlap")
            long_df["Referenzkunde_ID"] = int(ref_id)
            long_df["side"] = side
            rows.append(
                long_df[["Referenzkunde_ID", "side", "metric", "K", "overlap"]]
            )
    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame(
        columns=["Referenzkunde_ID", "side", "metric", "K", "overlap"]
    )


def compute_phase1_weighted_similarity(
    model,
    data_with_keys: pd.DataFrame,
    selected_customers: pd.DataFrame,
    weight_vectors: Dict[str, Iterable[float]],
    metrics_to_use: List[str] = ("proximity_avg",),
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Compute weighted similarities between reference customers and all others.

    The core similarity is the feature-wise weighted cosine average:
        S = sum_i w_i * cos(cust_i, ref_i) / sum_i w_i

    Args:
        model: Model providing vector representations.
        data_with_keys: DataFrame with customer data and keys.
        selected_customers: DataFrame of reference customers.
        weight_vectors: Dictionary mapping weight names to weight iterables.
        metrics_to_use: List of metric names to compute.
        verbose: If True, print progress information.

    Returns:
        DataFrame with similarity scores for each customer pair and weight/metric.

    Raises:
        ValueError: If input data is invalid.
    """
    if data_with_keys is None or len(data_with_keys) == 0:
        raise ValueError("data_with_keys is empty or None.")
    if selected_customers is None or len(selected_customers) == 0:
        raise ValueError("selected_customers is empty or None.")

    reference_ids = list(selected_customers.index)

    all_vectors = _compute_all_customer_vectors(model, data_with_keys)
    if not all_vectors or not all_vectors[0]:
        raise ValueError("No feature vectors found (all_vectors empty).")

    num_features = len(all_vectors[0])

    validated_weights: Dict[str, np.ndarray] = {}
    for weight_name, weight_values in weight_vectors.items():
        validated_weights[weight_name] = _validate_weight_vector(
            weight_values, num_features=num_features, weight_name=weight_name
        )

    metric_functions = {
        "proximity_avg": lambda a, b, w: weighted_proximity_avg(a, b, w),
    }

    rows = []
    total_refs = len(reference_ids)
    total_customers = len(data_with_keys)

    for ref_idx, ref_id in enumerate(reference_ids, start=1):
        if verbose:
            print(f"[Phase1] Ref {ref_idx}/{total_refs}: {ref_id}")

        ref_vectors = all_vectors[ref_id]
        if not ref_vectors or len(ref_vectors) != num_features:
            raise ValueError(
                f"Reference customer {ref_id} has no valid feature vectors."
            )

        for other_id in range(total_customers):
            other_vectors = all_vectors[other_id]
            if not other_vectors or len(other_vectors) != num_features:
                continue

            if other_id == ref_id:
                continue

            row = {"Referenzkunde_ID": int(ref_id), "Kunde_ID": int(other_id)}

            for weight_name, weights in validated_weights.items():
                for metric_name in metrics_to_use:
                    if metric_name not in metric_functions:
                        continue
                    col_name = (
                        f"S_{metric_name.replace('proximity_', '')}_{weight_name}"
                    )

                    try:
                        similarity = metric_functions[metric_name](
                            ref_vectors, other_vectors, weights
                        )
                        row[col_name] = float(similarity)
                    except Exception as e:
                        row[col_name] = np.nan
                        if verbose:
                            print(
                                f"  Warning: {metric_name}/{weight_name} "
                                f"for {ref_id}-{other_id} failed: {e}"
                            )

            rows.append(row)

    result_df = pd.DataFrame(rows)

    id_columns = ["Referenzkunde_ID", "Kunde_ID"]
    metric_columns = sorted(
        [c for c in result_df.columns if c not in id_columns]
    )
    return result_df[id_columns + metric_columns]


def create_rankings_with_scores(
    phase1_df: pd.DataFrame,
    top_n: int = 200,
    *,
    ref_col: str = "Referenzkunde_ID",
    id_col: str = "Kunde_ID",
    exclude_self_match: bool = True,
    verbose: bool = False,
) -> dict:
    """
    Generate top-N and bottom-N rankings for each metric and reference customer.

    Args:
        phase1_df: Similarity scores DataFrame from phase 1.
        top_n: Number of items to include in each ranking.
        ref_col: Name of reference customer ID column.
        id_col: Name of other customer ID column.
        exclude_self_match: If True, exclude self-comparisons.
        verbose: If True, print progress information.

    Returns:
        Dictionary with rankings per reference customer.
    """
    metric_columns = [
        c for c in phase1_df.columns if c not in [ref_col, id_col]
    ]

    reference_ids = phase1_df[ref_col].unique()
    rankings_dict = {}

    for ref_id in reference_ids:
        if verbose:
            print(f"Creating rankings for reference customer {ref_id}")

        ref_data = phase1_df[phase1_df[ref_col] == ref_id].copy()

        if exclude_self_match:
            ref_data = ref_data[ref_data[id_col] != ref_id]

        ref_dict = {"top": {}, "bottom": {}}

        for metric in metric_columns:
            top_sorted = ref_data.sort_values(by=metric, ascending=False).head(
                top_n
            )
            top_df = (
                top_sorted[[id_col, metric]]
                .rename(columns={metric: "score"})
                .copy()
            )
            top_df["rank"] = np.arange(1, len(top_df) + 1, dtype=int)
            ref_dict["top"][metric] = top_df.rename(columns={id_col: "Kunde_ID"})[
                ["Kunde_ID", "score", "rank"]
            ]

            bottom_sorted = ref_data.sort_values(by=metric, ascending=True).head(
                top_n
            )
            bottom_df = (
                bottom_sorted[[id_col, metric]]
                .rename(columns={metric: "score"})
                .copy()
            )
            bottom_df["rank"] = np.arange(1, len(bottom_df) + 1, dtype=int)
            ref_dict["bottom"][metric] = bottom_df.rename(
                columns={id_col: "Kunde_ID"}
            )[["Kunde_ID", "score", "rank"]]

        rankings_dict[int(ref_id)] = ref_dict

    return rankings_dict


def compute_overlap_curves_against_baseline(
    rankings_dict: dict,
    *,
    baseline_metric: str = "S_avg_average",
    top_n: int = 200,
    use_fraction: bool = False,
    verbose: bool = False,
) -> dict:
    """
    Compute overlap curves comparing all metrics against a baseline.

    For each K from 1 to top_n, calculates the overlap between the top-K sets
    of the baseline metric and each other metric.

    Args:
        rankings_dict: Rankings from create_rankings_with_scores.
        baseline_metric: Name of the metric to use as baseline.
        top_n: Maximum K to consider.
        use_fraction: If True, return overlap fraction instead of count.
        verbose: If True, print progress information.

    Returns:
        Dictionary with overlap curves per reference customer.

    Raises:
        ValueError: If baseline metric is missing.
    """
    result = {}

    for ref_id, pack in rankings_dict.items():
        if verbose:
            print(f"Computing overlap curves for reference customer {ref_id}")

        metrics = list(pack["top"].keys())
        if baseline_metric not in metrics:
            raise ValueError(
                f"Baseline '{baseline_metric}' missing for ref_id={ref_id}. "
                f"metrics={metrics}"
            )

        other_metrics = [m for m in metrics if m != baseline_metric]

        baseline_top_ids = (
            pack["top"][baseline_metric]["Kunde_ID"].astype(int).to_numpy()[:top_n]
        )
        baseline_bottom_ids = (
            pack["bottom"][baseline_metric]["Kunde_ID"]
            .astype(int)
            .to_numpy()[:top_n]
        )

        top_curves = {}
        bottom_curves = {}

        for metric in other_metrics:
            top_ids = (
                pack["top"][metric]["Kunde_ID"].astype(int).to_numpy()[:top_n]
            )
            bottom_ids = (
                pack["bottom"][metric]["Kunde_ID"].astype(int).to_numpy()[:top_n]
            )

            top_overlaps = np.empty(
                len(top_ids), dtype=float if use_fraction else int
            )
            bottom_overlaps = np.empty(
                len(bottom_ids), dtype=float if use_fraction else int
            )

            baseline_set, compare_set = set(), set()
            for i in range(len(top_ids)):
                baseline_set.add(int(baseline_top_ids[i]))
                compare_set.add(int(top_ids[i]))
                overlap = len(baseline_set.intersection(compare_set))
                top_overlaps[i] = overlap / (i + 1) if use_fraction else overlap

            baseline_set, compare_set = set(), set()
            for i in range(len(bottom_ids)):
                baseline_set.add(int(baseline_bottom_ids[i]))
                compare_set.add(int(bottom_ids[i]))
                overlap = len(baseline_set.intersection(compare_set))
                bottom_overlaps[i] = (
                    overlap / (i + 1) if use_fraction else overlap
                )

            top_curves[metric] = top_overlaps
            bottom_curves[metric] = bottom_overlaps

        top_df = pd.DataFrame(
            top_curves, index=np.arange(1, len(baseline_top_ids) + 1, dtype=int)
        )
        bottom_df = pd.DataFrame(
            bottom_curves,
            index=np.arange(1, len(baseline_bottom_ids) + 1, dtype=int),
        )
        top_df.index.name = "K"
        bottom_df.index.name = "K"

        result[int(ref_id)] = {
            "top_curves": top_df,
            "bottom_curves": bottom_df,
            "summary_stats": {
                "baseline_metric": baseline_metric,
                "top_n": int(top_n),
                "use_fraction": bool(use_fraction),
                "n_metrics_compared": len(other_metrics),
            },
        }

    return result


def compute_separation_scores(
    phase1_df: pd.DataFrame, rankings_dict: dict, top_n: int = 200
) -> dict:
    """
    Calculate separation scores (mean top similarity - mean bottom similarity).

    Args:
        phase1_df: Similarity scores DataFrame from phase 1.
        rankings_dict: Rankings from create_rankings_with_scores.
        top_n: Number of items to consider from each ranking.

    Returns:
        Dictionary with per-reference and summary DataFrames.

    Raises:
        ValueError: If inputs are invalid or no scores can be computed.
    """
    if phase1_df is None or len(phase1_df) == 0:
        raise ValueError("phase1_df is empty or None.")
    if rankings_dict is None or len(rankings_dict) == 0:
        raise ValueError("rankings_dict is empty or None.")

    metric_columns = [
        c
        for c in phase1_df.columns
        if c not in ["Referenzkunde_ID", "Kunde_ID"]
    ]
    if not metric_columns:
        raise ValueError("No metric columns found.")

    phase1_grouped = {}
    for ref_id in phase1_df["Referenzkunde_ID"].unique():
        subset = phase1_df[phase1_df["Referenzkunde_ID"] == ref_id].copy()
        phase1_grouped[int(ref_id)] = subset.set_index("Kunde_ID")

    def _extract_ids(data, n: int) -> list:
        """Extract customer IDs from various ranking formats."""
        if data is None:
            return []
        if isinstance(data, pd.DataFrame):
            if "Kunde_ID" not in data.columns:
                if data.index.name == "Kunde_ID":
                    return [int(v) for v in data.index.to_list()[:n]]
                raise ValueError(
                    f"Ranking DataFrame without Kunde_ID column. columns={data.columns.tolist()}"
                )
            return [int(v) for v in data["Kunde_ID"].to_list()[:n]]
        if isinstance(data, pd.Series):
            return [int(v) for v in data.to_list()[:n]]
        if isinstance(data, (list, tuple, np.ndarray)):
            return [int(v) for v in list(data)[:n]]
        raise TypeError(f"Unknown ranking type: {type(data)}")

    rows = []

    for ref_id, rank_data in rankings_dict.items():
        ref_id_int = int(ref_id)
        if ref_id_int not in phase1_grouped:
            continue

        subset = phase1_grouped[ref_id_int]
        top_dict = rank_data.get("top", {})
        bottom_dict = rank_data.get("bottom", {})

        metrics = list(top_dict.keys())

        for metric in metrics:
            if metric not in metric_columns:
                continue

            top_ids = _extract_ids(top_dict.get(metric), top_n)
            bottom_ids = _extract_ids(bottom_dict.get(metric), top_n)

            top_values = (
                pd.to_numeric(subset.reindex(top_ids)[metric], errors="coerce")
                .dropna()
            )
            bottom_values = (
                pd.to_numeric(subset.reindex(bottom_ids)[metric], errors="coerce")
                .dropna()
            )

            mean_top = (
                float(top_values.mean()) if len(top_values) > 0 else np.nan
            )
            mean_bottom = (
                float(bottom_values.mean()) if len(bottom_values) > 0 else np.nan
            )
            separation = (
                mean_top - mean_bottom
                if (np.isfinite(mean_top) and np.isfinite(mean_bottom))
                else np.nan
            )

            rows.append(
                {
                    "Referenzkunde_ID": ref_id_int,
                    "Metrik": metric,
                    "mean_top": mean_top,
                    "mean_bottom": mean_bottom,
                    "separation": separation,
                    "n_top_used": int(len(top_values)),
                    "n_bottom_used": int(len(bottom_values)),
                }
            )

    per_reference_df = pd.DataFrame(rows)
    if len(per_reference_df) == 0:
        raise ValueError(
            "No separation scores computed (check rankings/IDs/metric names)."
    )

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


def compute_featurewise_churn_propensity(
    model,
    data_with_keys: pd.DataFrame,
    weight_vectors: Dict[str, Iterable[float]],
    churn_col: str = "churn",
    include_churned: bool = False,
    zscore: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Compute churn propensity scores using feature-wise weighted cosine similarity.

    Score(customer) = sum_i w_i * cos(customer_i, churn_centroid_i) / sum_i w_i
    where churn_centroid_i is the mean vector of churned customers for feature i.

    Args:
        model: Model providing vector representations.
        data_with_keys: DataFrame with customer data and keys.
        weight_vectors: Dictionary mapping weight names to weight iterables.
        churn_col: Name of the churn indicator column.
        include_churned: If True, include churned customers in output.
        zscore: If True, add z-score normalized columns.
        verbose: If True, print summary statistics.

    Returns:
        DataFrame with churn propensity scores and ranks.

    Raises:
        ValueError: If churn column is missing or no churned customers exist.
    """
    if churn_col not in data_with_keys.columns:
        raise ValueError(f"Column '{churn_col}' missing in data_with_keys.")

    processed_df = data_with_keys.copy()
    churn_binary = _convert_to_binary_churn(processed_df[churn_col])
    processed_df[churn_col] = churn_binary

    churn_mask = churn_binary == 1
    active_mask = ~churn_mask
    if churn_mask.sum() == 0:
        raise ValueError("No churned customers found (churn=1).")

    all_vectors = _compute_all_customer_vectors(model, processed_df)
    if not all_vectors or not all_vectors[0]:
        raise ValueError("No feature vectors found.")
    num_features = len(all_vectors[0])

    result = pd.DataFrame(
        {
            "Kunde_ID": np.arange(len(processed_df), dtype=int),
            churn_col: churn_binary,
        }
    )

    churn_feature_centroids: List[np.ndarray] = []
    for feature_idx in range(num_features):
        feature_vectors = []
        for cust_idx in np.where(churn_mask)[0]:
            vec_list = all_vectors[cust_idx]
            if vec_list and len(vec_list) == num_features:
                feature_vectors.append(vec_list[feature_idx])
        if len(feature_vectors) == 0:
            example = next(
                (
                    v[feature_idx]
                    for v in all_vectors
                    if v and len(v) == num_features
                ),
                None,
            )
            if example is None:
                raise ValueError(
                    "Cannot determine dimension of feature vectors."
                )
            churn_feature_centroids.append(np.zeros_like(example))
        else:
            churn_feature_centroids.append(
                np.mean(np.stack(feature_vectors), axis=0)
            )

    for weight_name, weight_values in weight_vectors.items():
        weights = _validate_weight_vector(
            weight_values, num_features=num_features, weight_name=weight_name
        )

        scores = np.full(len(processed_df), np.nan, dtype=float)

        for cust_idx in range(len(processed_df)):
            vec_list = all_vectors[cust_idx]
            if not vec_list or len(vec_list) != num_features:
                continue

            score = weighted_proximity_avg(
                vec_list, churn_feature_centroids, weights
            )
            scores[cust_idx] = score

        if not include_churned:
            scores = np.where(active_mask, scores, np.nan)

        score_col = f"ChurnProp_{weight_name}"
        result[score_col] = scores

        rank_col = f"Rank_{weight_name}"
        result[rank_col] = result[score_col].rank(ascending=False, method="min")

        if zscore:
            z_col = f"ChurnPropZ_{weight_name}"
            result[z_col] = _zscore_normalize(result[score_col])

        if verbose:
            valid_scores = result[score_col].dropna()
            if len(valid_scores) > 0:
                print(
                    f"[ChurnFeaturewise] {weight_name}: "
                    f"min={float(valid_scores.min()):.4f} "
                    f"mean={float(valid_scores.mean()):.4f} "
                    f"median={float(valid_scores.median()):.4f} "
                    f"max={float(valid_scores.max()):.4f}"
                )

    return result


def compute_churn_propensity_scores(*args, **kwargs) -> pd.DataFrame:
    """
    Deprecated wrapper for compute_featurewise_churn_propensity.
    """
    return compute_featurewise_churn_propensity(*args, **kwargs)


def run_experiment2_weighting_analysis(
    model_list,
    data_with_keys: pd.DataFrame,
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
    verbose: bool = False,
) -> dict:
    """
    Execute experiment 2 comparing different weighting schemes.

    Args:
        model_list: List of (model, model_info) tuples.
        data_with_keys: DataFrame with customer data and keys.
        weight_vectors: Dictionary mapping weight names to weight iterables.
        metrics_to_use: List of metric names to compute.
        topn_n: Number of top/bottom customers for diversity selection.
        subset_k: Number of clusters for diversity selection.
        top_n: Number of items for rankings and overlap curves.
        base_dir: Directory for caching phase results.
        churn_col: Name of the churn indicator column.
        run_churn_propensity: If True, compute churn propensity scores.
        churn_include_churned: If True, include churned customers in churn analysis.
        churn_use_zscores: If True, add z-score normalized churn columns.
        verbose: If True, print progress information.

    Returns:
        Dictionary containing all experiment results.
    """
    results = {}

    selected_customers, cluster_centers, selected_cluster_info, kmeans = (
        get_diverse_customers_with_clusters(
            data_with_keys, k=5, verbose=verbose
        )
    )

    if verbose:
        print("\n" + "=" * 70)
        print("EXPERIMENT 2: AVERAGING vs. WEIGHTING ANALYSIS")
        print("=" * 70)
        print(f"\nWeight vectors: {list(weight_vectors.keys())}")
        print(f"Metrics: {list(metrics_to_use)}")

    for model_idx, (model, model_info) in enumerate(model_list):
        if verbose:
            print(f"\n{'='*60}\nMODEL {model_idx}: {model_info}\n{'='*60}\n")

        phase1_data = save_or_load_phase_results(
            phase_name="exp2_phase1_weighted_similarity",
            phase_data=None,
            base_dir=base_dir,
            model_idx=model_idx,
            model_info=model_info,
            selected_customers=selected_customers,
            verbose=verbose,
        )

        if phase1_data is None:
            phase1_df = compute_phase1_weighted_similarity(
                model=model,
                data_with_keys=data_with_keys,
                selected_customers=selected_customers,
                weight_vectors=weight_vectors,
                metrics_to_use=list(metrics_to_use),
                verbose=verbose,
            )
            phase1_df = save_or_load_phase_results(
                phase_name="exp2_phase1_weighted_similarity",
                phase_data=phase1_df,
                base_dir=base_dir,
                model_idx=model_idx,
                model_info=model_info,
                selected_customers=selected_customers,
                verbose=verbose,
            )
        else:
            phase1_df = phase1_data

        results[f"exp2_phase1_model{model_idx}"] = phase1_df

        phase2_rank_data = save_or_load_phase_results(
            phase_name="exp2_phase2_rankings_v2_scores",
            phase_data=None,
            base_dir=base_dir,
            model_idx=model_idx,
            model_info=model_info,
            selected_customers=selected_customers,
            verbose=verbose,
        )
        if phase2_rank_data is None:
            rankings = create_rankings_with_scores(
                phase1_df=phase1_df, top_n=top_n, verbose=verbose
            )
            rankings = save_or_load_phase_results(
                phase_name="exp2_phase2_rankings_v2_scores",
                phase_data=rankings,
                base_dir=base_dir,
                model_idx=model_idx,
                model_info=model_info,
                selected_customers=selected_customers,
                verbose=verbose,
            )
        else:
            rankings = phase2_rank_data
        results[f"exp2_phase2_rankings_model{model_idx}"] = rankings

        results[
            f"exp2_phase2_rankings_long_model{model_idx}"
        ] = rankings_to_long_dataframe(rankings)

        overlaps_data = save_or_load_phase_results(
            phase_name="exp2_phase2_overlaps_v2_curves",
            phase_data=None,
            base_dir=base_dir,
            model_idx=model_idx,
            model_info=model_info,
            selected_customers=selected_customers,
            verbose=verbose,
        )
        if overlaps_data is None:
            overlaps = compute_overlap_curves_against_baseline(
                rankings_dict=rankings,
                baseline_metric="S_avg_average",
                top_n=top_n,
                use_fraction=False,
                verbose=verbose,
            )
            overlaps = save_or_load_phase_results(
                phase_name="exp2_phase2_overlaps_v2_curves",
                phase_data=overlaps,
                base_dir=base_dir,
                model_idx=model_idx,
                model_info=model_info,
                selected_customers=selected_customers,
                verbose=verbose,
            )
            overlap_long = overlap_curves_to_long_dataframe(overlaps)
            overlap_agg = aggregate_overlap_long_df(overlap_long)
            phase3_summary_long = compute_phase3_summary_stats_long(phase1_df)
            phase3_hist_long = compute_phase3_hist_long(
                phase1_df=phase1_df, bins=50, use_global_bins_across_metrics=True
            )

        else:
            overlaps = overlaps_data
        results[f"exp2_phase2_overlaps_model{model_idx}"] = overlaps
        results[f"exp2_phase2_overlaps_long_model{model_idx}"] = overlap_long
        results[f"exp2_phase2_overlaps_agg_model{model_idx}"] = overlap_agg
        results[
            f"exp2_phase3_summary_long_model{model_idx}"
        ] = phase3_summary_long
        results[f"exp2_phase3_hist_long_model{model_idx}"] = phase3_hist_long

        results[
            f"exp2_phase2_overlaps_long_model{model_idx}"
        ] = overlap_curves_to_long_dataframe(overlaps)

        phase3_data = save_or_load_phase_results(
            phase_name="exp2_phase3_stats",
            phase_data=None,
            base_dir=base_dir,
            model_idx=model_idx,
            model_info=model_info,
            selected_customers=selected_customers,
            verbose=verbose,
        )
        if phase3_data is None:
            phase3 = compute_phase3_metric_statistics(
                phase1_df=phase1_df, bins=50, use_global_bins_across_metrics=True
            )
            phase3 = save_or_load_phase_results(
                phase_name="exp2_phase3_stats",
                phase_data=phase3,
                base_dir=base_dir,
                model_idx=model_idx,
                model_info=model_info,
                selected_customers=selected_customers,
                verbose=verbose,
            )
        else:
            phase3 = phase3_data
        results[f"exp2_phase3_stats_model{model_idx}"] = phase3

        phase4_data = save_or_load_phase_results(
            phase_name="exp2_phase4_separation",
            phase_data=None,
            base_dir=base_dir,
            model_idx=model_idx,
            model_info=model_info,
            selected_customers=selected_customers,
            verbose=verbose,
        )
        if phase4_data is None:
            phase4 = compute_separation_scores(
                phase1_df=phase1_df, rankings_dict=rankings, top_n=top_n
            )
            phase4 = save_or_load_phase_results(
                phase_name="exp2_phase4_separation",
                phase_data=phase4,
                base_dir=base_dir,
                model_idx=model_idx,
                model_info=model_info,
                selected_customers=selected_customers,
                verbose=verbose,
            )
        else:
            phase4 = phase4_data
        results[f"exp2_phase4_separation_model{model_idx}"] = phase4

        if run_churn_propensity:
            churn_phase_name = "exp2_phase3b_churn_propensity_featurewise"

            churn_data = save_or_load_phase_results(
                phase_name=churn_phase_name,
                phase_data=None,
                base_dir=base_dir,
                model_idx=model_idx,
                model_info=model_info,
                selected_customers=selected_customers,
                verbose=verbose,
            )
            if churn_data is None:
                churn_scores_df = compute_featurewise_churn_propensity(
                    model=model,
                    data_with_keys=data_with_keys,
                    weight_vectors=weight_vectors,
                    churn_col=churn_col,
                    include_churned=churn_include_churned,
                    zscore=churn_use_zscores,
                    verbose=verbose,
                )
                churn_scores_df = save_or_load_phase_results(
                    phase_name=churn_phase_name,
                    phase_data=churn_scores_df,
                    base_dir=base_dir,
                    model_idx=model_idx,
                    model_info=model_info,
                    selected_customers=selected_customers,
                    verbose=verbose,
                )
            else:
                churn_scores_df = churn_data

            results[f"exp2_churn_propensity_model{model_idx}"] = churn_scores_df

    return results


def aggregate_overlap_long_df(overlap_long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate overlap curves across reference customers.

    Args:
        overlap_long_df: Long-format overlap DataFrame.

    Returns:
        DataFrame with aggregated statistics per side, K, and metric.
    """
    if overlap_long_df is None or len(overlap_long_df) == 0:
        return pd.DataFrame(
            columns=[
                "side",
                "K",
                "metric",
                "overlap_mean",
                "overlap_median",
                "overlap_std",
                "n_refs",
            ]
        )

    agg = (
        overlap_long_df.groupby(["side", "K", "metric"], as_index=False)
        .agg(
            overlap_mean=("overlap", "mean"),
            overlap_median=("overlap", "median"),
            overlap_std=("overlap", "std"),
            n_refs=("Referenzkunde_ID", "nunique"),
        )
    )
    agg.sort_values(["side", "metric", "K"], inplace=True)
    return agg


def compute_phase3_summary_stats_long(
    phase1_df: pd.DataFrame,
    ref_col: str = "Referenzkunde_ID",
    id_col: str = "Kunde_ID",
) -> pd.DataFrame:
    """
    Compute per-reference summary statistics for each metric.

    Args:
        phase1_df: Similarity scores DataFrame.
        ref_col: Reference customer ID column name.
        id_col: Other customer ID column name.

    Returns:
        DataFrame with min, max, mean, std, negative fraction, and span.
    """
    metric_columns = [
        c for c in phase1_df.columns if c not in [ref_col, id_col]
    ]
    parts = []

    for metric in metric_columns:
        temp = phase1_df[[ref_col, metric]].copy()
        temp[metric] = pd.to_numeric(temp[metric], errors="coerce")
        grouped = temp.groupby(ref_col)[metric]

        stats = grouped.agg(["min", "max", "mean", "std"]).reset_index()
        stats.rename(columns={ref_col: "Referenzkunde_ID"}, inplace=True)
        stats["metric"] = metric

        negative = temp.assign(
            is_negative=lambda x: (x[metric] < 0).astype(int)
        ).groupby(ref_col)["is_negative"].mean().reset_index()
        negative.rename(
            columns={ref_col: "Referenzkunde_ID", "is_negative": "neg_frac"},
            inplace=True,
        )

        stats = stats.merge(negative, on="Referenzkunde_ID", how="left")
        stats["span"] = stats["max"] - stats["min"]

        parts.append(
            stats[
                [
                    "Referenzkunde_ID",
                    "metric",
                    "min",
                    "max",
                    "mean",
                    "std",
                    "neg_frac",
                    "span",
                ]
            ]
        )

    if parts:
        result = pd.concat(parts, ignore_index=True)
    else:
        result = pd.DataFrame(
            columns=[
                "Referenzkunde_ID",
                "metric",
                "min",
                "max",
                "mean",
                "std",
                "neg_frac",
                "span",
            ]
        )
    result.sort_values(["metric", "Referenzkunde_ID"], inplace=True)
    return result


def compute_phase3_hist_long(
    phase1_df: pd.DataFrame,
    bins: int = 50,
    use_global_bins_across_metrics: bool = True,
    ref_col: str = "Referenzkunde_ID",
    id_col: str = "Kunde_ID",
) -> pd.DataFrame:
    """
    Compute histogram data for each metric.

    Args:
        phase1_df: Similarity scores DataFrame.
        bins: Number of histogram bins.
        use_global_bins_across_metrics: If True, use same bins for all metrics.
        ref_col: Reference customer ID column name.
        id_col: Other customer ID column name.

    Returns:
        DataFrame with bin edges, centers, counts, and densities per metric.
    """
    metric_columns = [
        c for c in phase1_df.columns if c not in [ref_col, id_col]
    ]

    if use_global_bins_across_metrics:
        all_values = pd.concat(
            [pd.to_numeric(phase1_df[m], errors="coerce") for m in metric_columns],
            ignore_index=True,
        )
        all_values = all_values.dropna()
        if len(all_values) == 0:
            return pd.DataFrame(
                columns=[
                    "metric",
                    "bin_left",
                    "bin_right",
                    "bin_center",
                    "count",
                    "density",
                ]
            )
        global_min, global_max = float(all_values.min()), float(all_values.max())
        bin_edges = np.linspace(global_min, global_max, bins + 1)
    else:
        bin_edges = None

    rows = []
    for metric in metric_columns:
        values = (
            pd.to_numeric(phase1_df[metric], errors="coerce").dropna().to_numpy()
        )
        if len(values) == 0:
            continue

        if bin_edges is None:
            local_min, local_max = float(np.min(values)), float(np.max(values))
            current_edges = np.linspace(local_min, local_max, bins + 1)
        else:
            current_edges = bin_edges

        counts, edges = np.histogram(values, bins=current_edges)
        widths = np.diff(edges)
        density = counts / (counts.sum() * widths) if counts.sum() > 0 else np.nan

        for i in range(len(counts)):
            rows.append(
                {
                    "metric": metric,
                    "bin_left": float(edges[i]),
                    "bin_right": float(edges[i + 1]),
                    "bin_center": float((edges[i] + edges[i + 1]) / 2),
                    "count": int(counts[i]),
                    "density": float(density[i]) if counts.sum() > 0 else np.nan,
                }
            )

    result = pd.DataFrame(rows)
    if not result.empty:
        result.sort_values(["metric", "bin_center"], inplace=True)
    return result


# Deprecated functions kept for compatibility
def compute_overlap_curves_exp2(rankings_dict: dict, baseline_col: str = "S_avg_average", top_n: int = 200) -> dict:
    """Deprecated: use compute_overlap_curves_against_baseline."""
    print("Warning: compute_overlap_curves_exp2 is deprecated. Use compute_overlap_curves_against_baseline.")
    return compute_overlap_curves_against_baseline(
        rankings_dict=rankings_dict,
        baseline_metric=baseline_col,
        top_n=top_n,
        use_fraction=False,
        verbose=False,
    )


def create_rankings_for_phase2_exp2(phase1_df: pd.DataFrame, top_n: int = 200) -> dict:
    """Deprecated: use create_rankings_with_scores."""
    print("Warning: create_rankings_for_phase2_exp2 is deprecated. Use create_rankings_with_scores.")
    return create_rankings_with_scores(phase1_df=phase1_df, top_n=top_n)


def compute_churn_centroids_weighted(*args, **kwargs):
    """Deprecated function removed for lecture consistency."""
    raise NotImplementedError(
        "compute_churn_centroids_weighted is deprecated and removed. "
        "Use compute_featurewise_churn_propensity."
    )
