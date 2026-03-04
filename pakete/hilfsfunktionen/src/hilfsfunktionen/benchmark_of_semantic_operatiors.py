"""Benchmark module for evaluating semantic similarity operators on customer data.

This module provides comprehensive benchmarking functionality for comparing
different semantic similarity metrics on customer vector representations.
It includes phases for similarity calculation, ranking generation, overlap
analysis, statistical characterization, and separation score computation.
"""

import os
import time
from datetime import timedelta
from itertools import combinations
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .cosine_funktions import (
    proximity_max,
    proximity_avg,
    proximity_topn_avg,
    subset_proximity_avg,
    proximity_avg_all
)
from .key_vector_tests import get_key_values, compare_lists


def find_n_most_similar_to_key(model, key, method, n=5, df_with_keys=None):
    """Find the n most similar customers to a given key customer.
    
    Args:
        model: Word2Vec model containing vectors
        key: Index of the reference customer
        method: Similarity method to use
        n: Number of similar customers to return
        df_with_keys: DataFrame containing customer keys
        
    Returns:
        List of [customer_id, similarity_value] pairs sorted by similarity
    """
    vectors_key = []
    vectors_temp = []
    most_similar = []
    vectors_key = get_vector_list(model, key, df_with_keys=df_with_keys)
    for i in range(len(df_with_keys)):
        if i == key:
            continue
        vectors_temp = get_vector_list(model, i, df_with_keys=df_with_keys)
        match method:
            case "proximity_avg":
                avg_value = proximity_avg(vectors_key, vectors_temp)
            case "proximity_topn_avg":
                avg_value = proximity_topn_avg(vectors_key, vectors_temp)
            case "subset_proximity_avg":
                avg_value = subset_proximity_avg(vectors_key, vectors_temp)
            case "proximity_avg_all":
                avg_value = proximity_avg_all(vectors_key, vectors_temp)
            case _:
                return 'Unbekanntes Kommando'
        
        most_similar.append([i, avg_value])

    most_similar.sort(key=lambda x: x[1], reverse=True)
    return most_similar[:n]


def get_vector_list(model, key, df_with_keys=None):
    """Retrieve the list of vectors for a given customer key.
    
    Args:
        model: Word2Vec model containing vectors
        key: Customer index
        df_with_keys: DataFrame containing customer keys
        
    Returns:
        List of vectors for the customer
    """
    vector_list = []
    feature_names = []
    
    key_values = get_key_values(key, False, exclude_prefix="key", df_with_keys=df_with_keys)
    key_values_filtered = filter_keys(key_values)
    
    for value in key_values_filtered:
        vector_list.append(model.wv[value])
        feature_names.append(value)
    
    model.feature_names = feature_names
    return vector_list


def _compute_all_customer_vectors(model, df_with_keys):
    """Precompute vector lists for all customers to avoid repeated calculations.
    
    Args:
        model: Word2Vec model containing vectors
        df_with_keys: DataFrame containing customer keys
        
    Returns:
        List of vector lists for all customers
    """
    return [get_vector_list(model, i, df_with_keys=df_with_keys) for i in range(len(df_with_keys))]


def compute_phase1_similarity_df(
    model,
    df_with_keys: pd.DataFrame,
    selected_customers: pd.DataFrame,
    topn_values=(2, 5, 10),
    k_values=(3, 5, 7),
    verbose: bool = False
) -> pd.DataFrame:
    """Phase 1: Calculate similarity between reference customers and all others.
    
    Computes multiple similarity metrics for each pair:
        - S_max: ProximityMax
        - S_avg: ProximityAvg (cosine between centroids)
        - S_avgall: ProximityAvgAll (mean over all pairs)
        - S_topN_N: ProximityTopNAvg for N in topn_values
        - S_k_K: SubsetProximityAvg for K in k_values
    
    Args:
        model: Word2Vec model containing vectors
        df_with_keys: DataFrame with customer keys
        selected_customers: DataFrame of reference customers
        topn_values: Tuple of N values for top-N metrics
        k_values: Tuple of K values for subset metrics
        verbose: Whether to print progress information
        
    Returns:
        DataFrame with similarity values in long format
    """
    if df_with_keys is None or len(df_with_keys) == 0:
        raise ValueError("df_with_keys ist leer oder None.")

    if selected_customers is None or len(selected_customers) == 0:
        raise ValueError("selected_customers ist leer oder None.")

    reference_ids = list(selected_customers.index)
    all_vectors = _compute_all_customer_vectors(model, df_with_keys)

    rows = []
    total_refs = len(reference_ids)
    total_customers = len(df_with_keys)

    topn_values = tuple(int(n) for n in topn_values)
    k_values = tuple(int(k) for k in k_values)

    for r_idx, ref_id in enumerate(reference_ids, start=1):
        v_ref = all_vectors[ref_id]

        if verbose:
            print(f"[Phase 1] Referenz {r_idx}/{total_refs}: Kunde {ref_id} vs {total_customers-1} andere Kunden")

        for other_id in range(total_customers):
            if other_id == ref_id:
                continue

            v_other = all_vectors[other_id]

            row = {
                "Referenzkunde_ID": ref_id,
                "Kunde_ID": other_id,
                "S_max": proximity_max(v_ref, v_other),
                "S_avg": proximity_avg(v_ref, v_other),
                "S_avgall": proximity_avg_all(v_ref, v_other),
            }

            for n in topn_values:
                row[f"S_topN_{n}"] = proximity_topn_avg(v_ref, v_other, n=n)

            for k in k_values:
                try:
                    row[f"S_k_{k}"] = subset_proximity_avg(v_ref, v_other, subset_size=k)
                except ValueError:
                    row[f"S_k_{k}"] = np.nan

            rows.append(row)

    base_cols = ["Referenzkunde_ID", "Kunde_ID", "S_max", "S_avg", "S_avgall"]
    topn_cols = [f"S_topN_{n}" for n in topn_values]
    k_cols = [f"S_k_{k}" for k in k_values]
    cols = base_cols + topn_cols + k_cols

    return pd.DataFrame(rows)[cols]


def evaluate_most_similar_proximity_avg(model, test_cases, verbose, df_with_keys=None):
    """Evaluate proximity_avg similarity by comparing top matches with ground truth.
    
    Args:
        model: Word2Vec model
        test_cases: List of customer indices to test
        verbose: Whether to print detailed results
        df_with_keys: DataFrame with customer keys
        
    Returns:
        String containing evaluation results
    """
    most_similar = []
    result = "\n"
    for test_case in test_cases:
        result_text = f"Test für Key_{test_case}: \t"
        result += result_text
        if verbose:
            print(result_text)
        most_similar_sentences = find_n_most_similar_to_key(model, test_case, "proximity_avg", df_with_keys=df_with_keys)
        sentence_of_testcase = get_key_values(test_case, True, exclude_prefix="key", df_with_keys=df_with_keys)
        temp = 0
        for sentence in most_similar_sentences:
            output, anzahl_paare = compare_lists(get_key_values(sentence[0], True, exclude_prefix="key", df_with_keys=df_with_keys), sentence_of_testcase)
            temp += anzahl_paare
            if verbose:
                print(f"Key_{sentence[0]}: {output}")
        result += f"{temp} gleiche Werte von 55.\n"
    return result


def evaluate_most_similar_proximity_topn_avg(model, test_cases, verbose, df_with_keys=None, method="proximity_topn_avg"):
    """Evaluate proximity_topn_avg similarity by comparing top matches with ground truth.
    
    Args:
        model: Word2Vec model
        test_cases: List of customer indices to test
        verbose: Whether to print detailed results
        df_with_keys: DataFrame with customer keys
        method: Similarity method to use
        
    Returns:
        String containing evaluation results
    """
    most_similar = []
    result = "\n"
    for test_case in test_cases:
        result_text = f"Test für Key_{test_case}: \t"
        result += result_text
        if verbose:
            print(result_text)
        most_similar_sentences = find_n_most_similar_to_key(model, test_case, method, df_with_keys=df_with_keys)
        sentence_of_testcase = get_key_values(test_case, True, exclude_prefix="key", df_with_keys=df_with_keys)
        temp = 0
        for sentence in most_similar_sentences:
            output, anzahl_paare = compare_lists(get_key_values(sentence[0], True, exclude_prefix="key", df_with_keys=df_with_keys), sentence_of_testcase)
            temp += anzahl_paare
            if verbose:
                print(f"Key_{sentence[0]}: {output}")
        result += f"{temp} gleiche Werte von 55.\n"
    return result


def find_global_most_similar(model, verbose, n=5, df_with_keys=None):
    """Find the n most similar customer pairs globally using proximity_avg.
    
    Args:
        model: Word2Vec model
        verbose: Whether to print progress
        n: Number of top pairs to return
        df_with_keys: DataFrame with customer keys
        
    Returns:
        List of (i, j, similarity) for top n pairs
    """
    all_similarities = []
    total_cases = len(df_with_keys)
    total_iterations = total_cases * (total_cases - 1) // 2
    iterations_done = 0
    
    start_time = time.time()
    
    all_vectors = [get_vector_list(model, i, df_with_keys=df_with_keys) for i in range(total_cases)]
    
    for i in range(total_cases):
        for j in range(i + 1, total_cases):
            similarity = proximity_avg(all_vectors[i], all_vectors[j])
            all_similarities.append((i, j, similarity))
            
            iterations_done += 1
            
            if iterations_done % 1000000 == 0:
                elapsed = time.time() - start_time
                progress = iterations_done / total_iterations
                remaining = (elapsed / progress) - elapsed if progress > 0 else 0
                
                print(f"{iterations_done}/{total_iterations} - "
                      f"Vergangen: {timedelta(seconds=int(elapsed))} - "
                      f"Verbleibend: ~{timedelta(seconds=int(remaining))}")
    
    all_similarities.sort(key=lambda x: x[2], reverse=True)
    return all_similarities[:n]


def run_full_vector_test(model_list, verbose, very_verbose=False, 
                         df_with_keys=None, average_sentence_vectors=None,
                         base_dir="."):
    """Execute complete multi-phase benchmark test on given models.
    
    Args:
        model_list: List of (model, model_info) tuples
        verbose: Whether to print progress information
        very_verbose: Whether to print detailed information
        df_with_keys: DataFrame with customer keys
        average_sentence_vectors: Precomputed sentence vectors (unused)
        base_dir: Base directory for saving results
        
    Returns:
        Dictionary containing results from all phases
    """
    success = {}

    selected_customers, cluster_centers, selected_cluster_info, kmeans = get_diverse_customers_with_clusters(df_with_keys, k=5)

    if verbose:
        print("\nAusgewählte diverse Kunden für Tests:")
        print(selected_customers[['cluster_group', 'cluster_size'] + [col for col in selected_customers.columns if not col.startswith('key')]].to_string(index=False))
        interpret_clusters(selected_customers, cluster_centers, selected_cluster_info, df_with_keys, kmeans, verbose=verbose)

    for model_idx, (model, model_info) in enumerate(model_list):
        if verbose:
            print(f"\n{'='*60}")
            print(f"MODEL {model_idx}: {model_info}")
            print(f"{'='*60}\n")

        # Phase 1: Similarity calculation
        if verbose:
            print("PHASE 1: Ähnlichkeitsberechnung")
        
        phase1_data = save_or_load_phase_results(
            phase_name="phase1_similarity",
            phase_data=None,
            base_dir=base_dir,
            model_idx=model_idx,
            model_info=model_info,
            selected_customers=selected_customers,
            verbose=verbose
        )
        
        if phase1_data is None:
            phase1_df = compute_phase1_similarity_df(
                model=model,
                df_with_keys=df_with_keys,
                selected_customers=selected_customers,
                topn_values=(2, 5, 10),
                k_values=(3, 5, 7),
                verbose=verbose
            )
            
            phase1_df = save_or_load_phase_results(
                phase_name="phase1_similarity",
                phase_data=phase1_df,
                base_dir=base_dir,
                model_idx=model_idx,
                model_info=model_info,
                selected_customers=selected_customers,
                verbose=verbose
            )
        else:
            phase1_df = phase1_data
        
        success[f"phase1_model{model_idx}"] = phase1_df
        
        # Phase 2a: Rankings
        if verbose:
            print("\nPHASE 2a: Rankings")
        
        rankings_data = save_or_load_phase_results(
            phase_name="phase2_rankings",
            phase_data=None,
            base_dir=base_dir,
            model_idx=model_idx,
            model_info=model_info,
            selected_customers=selected_customers,
            verbose=verbose
        )
        
        if rankings_data is None:
            rankings = create_rankings_for_phase2(phase1_df, top_n=200, verbose=verbose)
            rankings = save_or_load_phase_results(
                phase_name="phase2_rankings",
                phase_data=rankings,
                base_dir=base_dir,
                model_idx=model_idx,
                model_info=model_info,
                selected_customers=selected_customers,
                verbose=verbose
            )
        else:
            rankings = rankings_data
        
        success[f"phase2_rankings_model{model_idx}"] = rankings
        
        # Phase 2b: Overlaps
        if verbose:
            print("\nPHASE 2b: Overlaps")
        
        overlaps_data = save_or_load_phase_results(
            phase_name="phase2_overlaps",
            phase_data=None,
            base_dir=base_dir,
            model_idx=model_idx,
            model_info=model_info,
            selected_customers=selected_customers,
            verbose=verbose
        )
        
        if overlaps_data is None:
            overlaps = compute_overlap_coefficients(rankings, verbose=verbose)
            overlaps = save_or_load_phase_results(
                phase_name="phase2_overlaps",
                phase_data=overlaps,
                base_dir=base_dir,
                model_idx=model_idx,
                model_info=model_info,
                selected_customers=selected_customers,
                verbose=verbose
            )
        else:
            overlaps = overlaps_data
        
        success[f"phase2_overlaps_model{model_idx}"] = overlaps
        
        if verbose:
            print(f"\nPhase 2 abgeschlossen für Model {model_idx}")

        # Phase 3: Statistical characterization
        if verbose:
            print("\nPHASE 3: Statistische Charakterisierung der Metriken")

        phase3_data = save_or_load_phase_results(
            phase_name="phase3_stats",
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
                phase_name="phase3_stats",
                phase_data=phase3,
                base_dir=base_dir,
                model_idx=model_idx,
                model_info=model_info,
                selected_customers=selected_customers,
                verbose=verbose
            )
        else:
            phase3 = phase3_data

        success[f"phase3_stats_model{model_idx}"] = phase3

        # Phase 4: Discrimination ability (Separation Score)
        if verbose:
            print("\nPHASE 4: Diskriminierungsfähigkeit (Separation Score)")

        phase4_data = save_or_load_phase_results(
            phase_name="phase4_separation",
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
                top_n=200
            )
            phase4 = save_or_load_phase_results(
                phase_name="phase4_separation",
                phase_data=phase4,
                base_dir=base_dir,
                model_idx=model_idx,
                model_info=model_info,
                selected_customers=selected_customers,
                verbose=verbose
            )
        else:
            phase4 = phase4_data

        success[f"phase4_separation_model{model_idx}"] = phase4
    
    return success


def create_rankings_for_phase2(phase1_df, top_n=200, verbose=False):
    """Create top and bottom rankings for each metric per reference customer.
    
    Args:
        phase1_df: DataFrame from phase 1 with similarity values
        top_n: Number of items to include in top/bottom rankings
        verbose: Whether to print progress information
        
    Returns:
        Dictionary with rankings structured by reference customer
    """
    metric_cols = [col for col in phase1_df.columns 
                   if col not in ['Referenzkunde_ID', 'Kunde_ID']]
    
    reference_ids = phase1_df['Referenzkunde_ID'].unique()
    
    rankings_dict = {}
    
    for ref_id in reference_ids:
        if verbose:
            print(f"Erstelle Rankings für Referenzkunde {ref_id}")
        
        ref_data = phase1_df[phase1_df['Referenzkunde_ID'] == ref_id].copy()
        
        ref_dict = {'top': {}, 'bottom': {}}
        
        for metric in metric_cols:
            bottom_sorted = ref_data.sort_values(by=metric, ascending=True)
            bottom_ranking = bottom_sorted['Kunde_ID'].head(top_n).tolist()
            
            top_sorted = ref_data.sort_values(by=metric, ascending=False)
            top_ranking = top_sorted['Kunde_ID'].head(top_n).tolist()
            
            ref_dict['top'][metric] = top_ranking
            ref_dict['bottom'][metric] = bottom_ranking
        
        rankings_dict[ref_id] = ref_dict
    
    return rankings_dict


def compute_overlap_coefficients(rankings_dict, verbose=False):
    """Calculate overlap coefficients between all metric pairs.
    
    Args:
        rankings_dict: Dictionary from create_rankings_for_phase2()
        verbose: Whether to print progress information
        
    Returns:
        Dictionary with overlap coefficients for each reference customer
    """
    results_dict = {}
    
    for ref_id, rankings in rankings_dict.items():
        if verbose:
            print(f"Berechne Overlaps für Referenzkunde {ref_id}")
        
        metrics = list(rankings['top'].keys())
        
        top_overlap_df = pd.DataFrame(index=metrics, columns=metrics)
        bottom_overlap_df = pd.DataFrame(index=metrics, columns=metrics)
        
        for metric1, metric2 in combinations(metrics, 2):
            top_set1 = set(rankings['top'][metric1])
            top_set2 = set(rankings['top'][metric2])
            top_overlap = len(top_set1.intersection(top_set2)) / min(len(top_set1), len(top_set2))
            
            bottom_set1 = set(rankings['bottom'][metric1])
            bottom_set2 = set(rankings['bottom'][metric2])
            bottom_overlap = len(bottom_set1.intersection(bottom_set2)) / min(len(bottom_set1), len(bottom_set2))
            
            top_overlap_df.loc[metric1, metric2] = top_overlap
            top_overlap_df.loc[metric2, metric1] = top_overlap
            bottom_overlap_df.loc[metric1, metric2] = bottom_overlap
            bottom_overlap_df.loc[metric2, metric1] = bottom_overlap
        
        for metric in metrics:
            top_overlap_df.loc[metric, metric] = 1.0
            bottom_overlap_df.loc[metric, metric] = 1.0
        
        summary_rows = []
        for metric in metrics:
            top_avg = top_overlap_df.loc[metric, top_overlap_df.columns != metric].mean()
            bottom_avg = bottom_overlap_df.loc[metric, bottom_overlap_df.columns != metric].mean()
            
            summary_rows.append({
                'Metrik': metric,
                'Top_Avg_Overlap': top_avg,
                'Bottom_Avg_Overlap': bottom_avg,
                'Overlap_Diff': top_avg - bottom_avg
            })
        
        summary_df = pd.DataFrame(summary_rows)
        
        results_dict[ref_id] = {
            'top_overlaps': top_overlap_df,
            'bottom_overlaps': bottom_overlap_df,
            'summary_stats': summary_df
        }
        
        if verbose:
            print(f"  Top-Overlap-Matrix:\n{top_overlap_df.round(3)}\n")
            print(f"  Zusammenfassung:\n{summary_df.round(3)}\n")
    
    return results_dict


def get_diverse_customers_with_clusters(df, k=5, random_state=42, verbose=False):
    """Select diverse customers by clustering and picking representatives.
    
    Args:
        df: DataFrame with customer data
        k: Number of clusters
        random_state: Random seed for reproducibility
        verbose: Whether to print progress information
        
    Returns:
        Tuple of (selected_customers, cluster_centers, cluster_info, kmeans_model)
    """
    df_encoded = prepare_data_for_clustering(df, verbose)
    
    features = [col for col in df_encoded.columns if col not in ['key_1', 'key_2', 'key3', 'key4', 'key5', 'key6']]
    X = df_encoded[features].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    
    selected_indices = []
    selected_cluster_info = []
    
    for cluster_id in range(k):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        
        if len(cluster_indices) > 0:
            cluster_points = X_scaled[cluster_indices]
            cluster_center = kmeans.cluster_centers_[cluster_id]
            
            distances = np.linalg.norm(cluster_points - cluster_center, axis=1)
            
            farthest_idx_in_cluster = np.argmax(distances)
            selected_idx = cluster_indices[farthest_idx_in_cluster]
            
            selected_indices.append(selected_idx)
            
            cluster_info = {
                'cluster_id': cluster_id,
                'cluster_size': len(cluster_indices),
                'cluster_center_original': cluster_centers[cluster_id],
                'selected_customer_idx': selected_idx
            }
            selected_cluster_info.append(cluster_info)
    
    selected_customers = df.iloc[selected_indices].copy()
    
    selected_customers['cluster_group'] = [info['cluster_id'] for info in selected_cluster_info]
    selected_customers['cluster_size'] = [info['cluster_size'] for info in selected_cluster_info]
    
    return selected_customers, cluster_centers, selected_cluster_info, kmeans


def prepare_data_for_clustering(df, verbose=False):
    """Prepare data for clustering by encoding categorical variables.
    
    Args:
        df: Original DataFrame
        verbose: Whether to print progress information
        
    Returns:
        DataFrame with encoded features suitable for clustering
    """
    df_clean = df.copy()
    
    if verbose:
        print("Schritt 1: Redundante Spalten entfernen...")
    
    key_patterns = ['key_1', 'key_2', 'key3', 'key4', 'key5', 'key6']
    cols_to_drop = []
    
    for col in df_clean.columns:
        if any(pattern in col for pattern in key_patterns):
            cols_to_drop.append(col)
    
    if verbose:
        print(f"Entferne {len(cols_to_drop)} Schlüsselspalten")
    df_clean = df_clean.drop(columns=cols_to_drop)
    
    if verbose:
        print("\nSchritt 2: Kategorische Spalten identifizieren...")
    
    categorical_columns = [
        'Credit_Score', 'Country', 'gender', 'age', 'tenure',
        'balance', 'products_number', 'credit_card', 'active_member',
        'estimated_salary', 'churn'
    ]
    
    actual_categorical_cols = []
    for pattern in categorical_columns:
        matching = [col for col in df_clean.columns if pattern.lower() in col.lower()]
        actual_categorical_cols.extend(matching)
    
    if verbose:
        print(f"Gefundene kategorische Spalten: {actual_categorical_cols}")
    
    if verbose:
        print("\nSchritt 3: One-Hot Encoding durchführen...")
    
    encoded_dfs = []
    
    for col in actual_categorical_cols:
        unique_vals = df_clean[col].astype(str).unique()
        
        if len(unique_vals) == 2:
            mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
            encoded = df_clean[col].map(mapping)
            encoded_dfs.append(pd.DataFrame({col: encoded}))
        else:
            base_name = col.split('_')[0] if '_' in col else col
            
            if df_clean[col].astype(str).str.contains('_').any():
                dummies = pd.get_dummies(df_clean[col], prefix=base_name)
                encoded_dfs.append(dummies)
            else:
                dummies = pd.get_dummies(df_clean[col], prefix=col)
                encoded_dfs.append(dummies)
    
    numeric_cols = []
    for col in df_clean.columns:
        if col not in actual_categorical_cols:
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                numeric_cols.append(col)
    
    if verbose:
        print(f"Numerische Spalten: {numeric_cols}")
    
    if numeric_cols:
        encoded_dfs.append(df_clean[numeric_cols])
    
    result = pd.concat(encoded_dfs, axis=1)
    
    if verbose:
        print(f"\nErgebnis: {result.shape[1]} Features nach Encoding")
    
    return result


def interpret_clusters(selected_customers, cluster_centers, selected_cluster_info, 
                       df_original, kmeans, feature_names=None, verbose=True):
    """Interpret cluster characteristics and selected representatives.
    
    Args:
        selected_customers: DataFrame of selected customers
        cluster_centers: Cluster centers in original feature space
        selected_cluster_info: Information about each cluster
        df_original: Original DataFrame before encoding
        kmeans: Trained KMeans model
        feature_names: Names of features after encoding
        verbose: Whether to print interpretation
        
    Returns:
        Dictionary with cluster interpretation data
    """
    if verbose:
        print("=" * 70)
        print("CLUSTER-INTERPRETATION")
        print("=" * 70)
    
    cluster_summary = []
    
    for info in selected_cluster_info:
        cluster_id = info['cluster_id']
        cluster_size = info['cluster_size']
        center = info['cluster_center_original']
        
        cluster_customer = selected_customers[selected_customers['cluster_group'] == cluster_id]
        
        if not cluster_customer.empty:
            customer_idx = cluster_customer.index[0]
            
            summary = {
                'cluster_id': cluster_id,
                'size': cluster_size,
                'percentage': (cluster_size / len(df_original)) * 100,
                'customer_key': cluster_customer.iloc[0].get('key_1', f'Kunde_{customer_idx}'),
                'center_values': center
            }
            cluster_summary.append(summary)
    
    if feature_names is None:
        encoded_cols = []
        for col in selected_customers.columns:
            if 'cluster' not in col.lower() and 'key' not in col.lower():
                encoded_cols.append(col)
        
        feature_groups = {}
        for col in encoded_cols:
            if '_' in col:
                base = col.split('_')[0]
                if base not in feature_groups:
                    feature_groups[base] = []
                feature_groups[base].append(col)
        
        feature_names = []
        for base, cols in feature_groups.items():
            if len(cols) > 1:
                feature_names.extend(cols)
            else:
                feature_names.append(cols[0])
    
    if verbose:
        print(f"\nGefundene {len(cluster_summary)} Cluster:")
        print("-" * 70)
    
    detailed_interpretations = []
    
    for summary in cluster_summary:
        cluster_id = summary['cluster_id']
        
        if verbose:
            print(f"\n📊 CLUSTER {cluster_id}")
            print(f"   Größe: {summary['size']} Kunden ({summary['percentage']:.1f}%)")
            print(f"   Repräsentant: {summary['customer_key']}")
        
        center = summary['center_values']
        
        if len(center) > 0:
            abs_center = np.abs(center)
            
            high_value_indices = np.where(center > 0.5)[0]
            
            if len(high_value_indices) > 0:
                sorted_indices = high_value_indices[np.argsort(-center[high_value_indices])]
                
                top_indices = sorted_indices[:5]
                
                interpretation = {
                    'cluster_id': cluster_id,
                    'top_features': [],
                    'characteristics': []
                }
                
                if verbose:
                    print(f"   Charakteristische Merkmale:")
        
        cluster_customers = df_original.iloc[
            np.where(kmeans.labels_ == cluster_id)[0]
        ]
        
        if verbose and len(cluster_customers) > 0:
            important_cols = ['Credit_Score', 'Country', 'age', 'estimated_salary', 'balance']
            available_cols = [col for col in important_cols if col in cluster_customers.columns]
            
            print(f"   Typische Werte in dieser Gruppe:")
            for col in available_cols[:3]:
                if col in cluster_customers.columns:
                    most_common = cluster_customers[col].mode()
                    if not most_common.empty:
                        value = most_common.iloc[0]
                        if isinstance(value, str) and len(value) > 30:
                            value = value[:27] + "..."
                        print(f"   • {col}: {value}")
    
    return {
        'cluster_summary': cluster_summary,
        'detailed_interpretations': detailed_interpretations,
        'feature_names': feature_names
    }


def save_or_load_phase_results(phase_name, phase_data, base_dir, model_idx, 
                               model_info, selected_customers, verbose=False):
    """Save or load phase results using NumPy .npz format.
    
    Args:
        phase_name: Name of the phase
        phase_data: Data to save (None if loading)
        base_dir: Base directory
        model_idx: Index of current model
        model_info: Model information dictionary
        selected_customers: DataFrame with selected customers
        verbose: Whether to print status messages
        
    Returns:
        Loaded data or newly saved data
    """
    customer_ids = "_".join(str(idx) for idx in selected_customers.index.tolist())
    
    filename = f"{base_dir}/precalculated_benchmark_values/{phase_name}_"
    
    for key, value in model_info.items():
        if isinstance(value, (str, int, float)):
            filename += f"_{key[:3]}_{value}"
    
    filename += f"_model{model_idx}_customers_{customer_ids}.npz"
    
    if os.path.exists(filename):
        if verbose:
            print(f"Lade {phase_name} von {filename}")
        
        loaded_data = np.load(filename, allow_pickle=True)
        
        if phase_name == "phase1_similarity":
            df = pd.DataFrame({
                'Referenzkunde_ID': loaded_data['referenz_ids'],
                'Kunde_ID': loaded_data['kunde_ids']
            })
            
            metric_names = loaded_data['metric_names']
            for i, metric_name in enumerate(metric_names):
                df[metric_name] = loaded_data[f'metric_{i}']
            
            return df
        elif phase_name in ("phase3_stats", "phase4_separation"):
            loaded_obj = loaded_data["payload"].item()
            return loaded_obj
        
        elif phase_name == "phase2_rankings":
            rankings_dict = {}
            ref_ids = loaded_data['ref_ids']
            
            for ref_id in ref_ids:
                ref_id_str = str(ref_id)
                rankings_dict[ref_id] = {
                    'top': {},
                    'bottom': {}
                }
                
                metrics = loaded_data[f'{ref_id_str}_metrics']
                
                for metric in metrics:
                    top_key = f'{ref_id_str}_top_{metric}'
                    if top_key in loaded_data:
                        rankings_dict[ref_id]['top'][metric] = loaded_data[top_key].tolist()
                    
                    bottom_key = f'{ref_id_str}_bottom_{metric}'
                    if bottom_key in loaded_data:
                        rankings_dict[ref_id]['bottom'][metric] = loaded_data[bottom_key].tolist()
            
            return rankings_dict
        
        elif phase_name == "phase2_overlaps":
            overlaps_dict = {}
            ref_ids = loaded_data['ref_ids']
            
            for ref_id in ref_ids:
                ref_id_str = str(ref_id)
                ref_key = f'{ref_id_str}_data'
                
                if ref_key in loaded_data:
                    ref_data = loaded_data[ref_key].item()
                    overlaps_dict[ref_id] = ref_data
            
            return overlaps_dict
        elif phase_name in ("phase3_stats", "phase4_separation"):
            save_dict = {
                "payload": np.array([phase_data], dtype=object)
            }
            np.savez(filename, **save_dict, allow_pickle=True)
    
    if phase_data is None:
        return None
    
    if verbose:
        print(f"Speichere {phase_name} in {filename}")
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    if phase_name == "phase1_similarity" and isinstance(phase_data, pd.DataFrame):
        save_dict = {
            'referenz_ids': phase_data['Referenzkunde_ID'].to_numpy(),
            'kunde_ids': phase_data['Kunde_ID'].to_numpy()
        }
        
        metric_cols = [col for col in phase_data.columns 
                      if col not in ['Referenzkunde_ID', 'Kunde_ID']]
        save_dict['metric_names'] = np.array(metric_cols, dtype=object)
        
        for i, metric in enumerate(metric_cols):
            save_dict[f'metric_{i}'] = phase_data[metric].to_numpy()
        
        np.savez(filename, **save_dict)
        
    elif phase_name == "phase2_rankings" and isinstance(phase_data, dict):
        save_dict = {
            'ref_ids': np.array(list(phase_data.keys()))
        }
        
        for ref_id, ref_data in phase_data.items():
            ref_id_str = str(ref_id)
            metrics = list(ref_data['top'].keys())
            save_dict[f'{ref_id_str}_metrics'] = np.array(metrics, dtype=object)
            
            for metric in metrics:
                save_dict[f'{ref_id_str}_top_{metric}'] = np.array(ref_data['top'][metric])
                save_dict[f'{ref_id_str}_bottom_{metric}'] = np.array(ref_data['bottom'][metric])
        
        np.savez(filename, **save_dict)
        
    elif phase_name == "phase2_overlaps" and isinstance(phase_data, dict):
        save_dict = {
            'ref_ids': np.array(list(phase_data.keys()))
        }
        
        for ref_id, ref_data in phase_data.items():
            ref_id_str = str(ref_id)
            save_dict[f'{ref_id_str}_data'] = np.array([ref_data], dtype=object)
        
        np.savez(filename, **save_dict, allow_pickle=True)
    
    return phase_data


def compute_phase3_metric_statistics(
    phase1_df: pd.DataFrame,
    bins: int = 50,
    use_global_bins_across_metrics: bool = True
) -> dict:
    """Phase 3: Statistical characterization of metrics.
    
    Computes for each metric:
        - mean, median, std, min, max, range
        - count_negative, count_total, share_negative
        - histogram (counts + bin_edges)
    
    Args:
        phase1_df: DataFrame from phase 1
        bins: Number of histogram bins
        use_global_bins_across_metrics: Whether to use common bin edges
        
    Returns:
        Dictionary with statistics and histograms
    """
    if phase1_df is None or len(phase1_df) == 0:
        raise ValueError("phase1_df ist leer oder None.")

    metric_cols = [c for c in phase1_df.columns if c not in ["Referenzkunde_ID", "Kunde_ID"]]
    if not metric_cols:
        raise ValueError("Keine Metrikspalten gefunden (erwartet: alles außer Referenzkunde_ID, Kunde_ID).")

    all_values = []
    for m in metric_cols:
        s = pd.to_numeric(phase1_df[m], errors="coerce").dropna()
        if len(s) > 0:
            all_values.append(s.to_numpy())
    if not all_values:
        raise ValueError("Alle Metrikspalten enthalten nur NaN/nicht-numerische Werte.")

    all_values = np.concatenate(all_values)

    if use_global_bins_across_metrics:
        global_min = np.nanmin(all_values)
        global_max = np.nanmax(all_values)
        if np.isclose(global_min, global_max):
            bin_edges_global = np.linspace(global_min - 1e-9, global_max + 1e-9, bins + 1)
        else:
            bin_edges_global = np.linspace(global_min, global_max, bins + 1)
    else:
        bin_edges_global = None

    rows = []
    histograms = {}

    for m in metric_cols:
        s = pd.to_numeric(phase1_df[m], errors="coerce")
        s_clean = s.dropna()

        if len(s_clean) == 0:
            rows.append({
                "Metrik": m,
                "count_total": 0,
                "mean": np.nan,
                "median": np.nan,
                "std": np.nan,
                "min": np.nan,
                "max": np.nan,
                "range": np.nan,
                "count_negative": 0,
                "share_negative": np.nan,
            })
            histograms[m] = {"bin_edges": (bin_edges_global if bin_edges_global is not None else np.array([])),
                             "counts": np.zeros(bins, dtype=int)}
            continue

        arr = s_clean.to_numpy()
        m_mean = float(np.mean(arr))
        m_median = float(np.median(arr))
        m_std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        m_min = float(np.min(arr))
        m_max = float(np.max(arr))
        m_range = m_max - m_min

        count_negative = int(np.sum(arr < 0))
        count_total = int(len(arr))
        share_negative = count_negative / count_total if count_total > 0 else np.nan

        if use_global_bins_across_metrics:
            counts, bin_edges = np.histogram(arr, bins=bin_edges_global)
        else:
            if np.isclose(m_min, m_max):
                bin_edges = np.linspace(m_min - 1e-9, m_max + 1e-9, bins + 1)
            else:
                bin_edges = np.linspace(m_min, m_max, bins + 1)
            counts, bin_edges = np.histogram(arr, bins=bin_edges)

        histograms[m] = {"bin_edges": bin_edges, "counts": counts}

        rows.append({
            "Metrik": m,
            "count_total": count_total,
            "mean": m_mean,
            "median": m_median,
            "std": m_std,
            "min": m_min,
            "max": m_max,
            "range": m_range,
            "count_negative": count_negative,
            "share_negative": share_negative,
        })

    stats_df = pd.DataFrame(rows).sort_values("Metrik").reset_index(drop=True)

    return {
        "stats_df": stats_df,
        "histograms": histograms,
        "bins": int(bins),
        "use_global_bins_across_metrics": bool(use_global_bins_across_metrics),
    }


def compute_phase4_separation_scores(
    phase1_df: pd.DataFrame,
    rankings_dict: dict,
    top_n: int = 200
) -> dict:
    """Phase 4: Compute separation scores for discrimination ability.
    
    For each metric and reference customer:
        Separation = mean(similarity(top)) - mean(similarity(bottom))
    
    Args:
        phase1_df: DataFrame from phase 1
        rankings_dict: Rankings from create_rankings_for_phase2()
        top_n: Number of items in rankings
        
    Returns:
        Dictionary with per-reference and summary DataFrames
    """
    if phase1_df is None or len(phase1_df) == 0:
        raise ValueError("phase1_df ist leer oder None.")
    if rankings_dict is None or len(rankings_dict) == 0:
        raise ValueError("rankings_dict ist leer oder None.")

    metric_cols = [c for c in phase1_df.columns if c not in ["Referenzkunde_ID", "Kunde_ID"]]
    if not metric_cols:
        raise ValueError("Keine Metrikspalten gefunden (erwartet: alles außer Referenzkunde_ID, Kunde_ID).")

    phase1_grouped = {}
    for ref_id in phase1_df["Referenzkunde_ID"].unique():
        sub = phase1_df[phase1_df["Referenzkunde_ID"] == ref_id].copy()
        sub = sub.set_index("Kunde_ID")
        phase1_grouped[ref_id] = sub

    rows = []

    for ref_id, rdata in rankings_dict.items():
        if ref_id not in phase1_grouped:
            continue

        sub = phase1_grouped[ref_id]
        top_dict = rdata.get("top", {})
        bottom_dict = rdata.get("bottom", {})

        metrics = list(top_dict.keys())

        for m in metrics:
            if m not in metric_cols:
                continue

            top_ids = top_dict.get(m, [])[:top_n]
            bottom_ids = bottom_dict.get(m, [])[:top_n]

            top_vals = pd.to_numeric(sub.reindex(top_ids)[m], errors="coerce").dropna()
            bottom_vals = pd.to_numeric(sub.reindex(bottom_ids)[m], errors="coerce").dropna()

            mean_top = float(top_vals.mean()) if len(top_vals) > 0 else np.nan
            mean_bottom = float(bottom_vals.mean()) if len(bottom_vals) > 0 else np.nan
            separation = mean_top - mean_bottom if (np.isfinite(mean_top) and np.isfinite(mean_bottom)) else np.nan

            rows.append({
                "Referenzkunde_ID": ref_id,
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


def filter_keys(values):
    """Filter out key_ values but keep the first occurrence.
    
    Args:
        values: List of strings
        
    Returns:
        Filtered list
    """
    filtered = []
    first_key_found = False
    
    for value in values:
        if value.startswith('key_'):
            if not first_key_found:
                filtered.append(value)
                first_key_found = True
        else:
            filtered.append(value)
    
    return filtered
