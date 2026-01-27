from .cosine_funktions import (
    weighted_proximity_max,
    weighted_proximity_avg,
    weighted_proximity_topn_avg,
    weighted_subset_proximity_avg,
    weighted_proximity_avg_all
)

from .benchmark_of_semantic_operatiors import (
    _compute_all_customer_vectors,
    get_diverse_customers_with_clusters,
    save_or_load_phase_results,
    create_rankings_for_phase2,
    compute_overlap_coefficients,
    compute_phase3_metric_statistics,
    compute_phase4_separation_scores
)

import os
import time
import pandas as pd
import numpy as np
from typing import List, Tuple, Iterable


def compute_phase1_weighted_similarity_df(
    model,
    df_with_keys: pd.DataFrame,
    selected_customers: pd.DataFrame,
    weight_vectors: dict,
    metrics_to_use: List[str] = ['proximity_avg', 'proximity_topn_avg'],
    topn_n: int = 3,
    subset_k: int = 3,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Phase 1 für Experiment 2: Gewichtete Ähnlichkeitsberechnung.
    
    Parameters:
    -----------
    model : Word2Vec model
        Trainiertes Word2Vec-Modell
    df_with_keys : pd.DataFrame
        DataFrame mit allen Kunden
    selected_customers : pd.DataFrame
        Referenzkunden
    weight_vectors : dict
        Dictionary mit Gewichtungsvektoren, z.B.:
        {'average': [1,1,1,...], 'weighted_base': [0,1,3,...], ...}
    metrics_to_use : List[str]
        Liste der zu verwendenden Basis-Metriken
        Optionen: 'proximity_avg', 'proximity_topn_avg', 'proximity_max', 
                  'proximity_avg_all', 'subset_proximity_avg'
    topn_n : int
        N-Parameter für proximity_topn_avg
    subset_k : int
        k-Parameter für subset_proximity_avg
    verbose : bool
        Fortschrittsanzeige
    
    Returns:
    --------
    pd.DataFrame mit Spalten:
        Referenzkunde_ID, Kunde_ID, S_avg_average, S_avg_weighted_base, ...
    """
    if df_with_keys is None or len(df_with_keys) == 0:
        raise ValueError("df_with_keys ist leer oder None.")
    
    if selected_customers is None or len(selected_customers) == 0:
        raise ValueError("selected_customers ist leer oder None.")
    
    reference_ids = list(selected_customers.index)
    
    # 1) Precompute: Vektorliste für alle Kunden (WIEDERVERWENDET!)
    all_vectors = _compute_all_customer_vectors(model, df_with_keys)
    
    rows = []
    total_refs = len(reference_ids)
    total_customers = len(df_with_keys)
    
    # Mapping von Metrik-Namen zu Funktionen
    metric_functions = {
        'proximity_avg': weighted_proximity_avg,
        'proximity_topn_avg': lambda v1, v2, w: weighted_proximity_topn_avg(v1, v2, w, n=topn_n),
        'proximity_max': weighted_proximity_max,
        'proximity_avg_all': weighted_proximity_avg_all,
        'subset_proximity_avg': lambda v1, v2, w: weighted_subset_proximity_avg(v1, v2, w, subset_size=subset_k)
    }
    
    for r_idx, ref_id in enumerate(reference_ids, start=1):
        v_ref = all_vectors[ref_id]
        
        if verbose:
            print(f"[Phase 1 Weighted] Referenz {r_idx}/{total_refs}: Kunde {ref_id} vs {total_customers-1} andere Kunden")
        
        for other_id in range(total_customers):
            if other_id == ref_id:
                continue
            
            v_other = all_vectors[other_id]
            
            row = {
                "Referenzkunde_ID": ref_id,
                "Kunde_ID": other_id
            }
            
            # Für jede Gewichtung und jede Metrik berechnen
            for weight_name, weight_vec in weight_vectors.items():
                weight_array = np.array(weight_vec, dtype=float)
                
                for metric_name in metrics_to_use:
                    if metric_name not in metric_functions:
                        continue
                    
                    try:
                        metric_func = metric_functions[metric_name]
                        sim = metric_func(v_ref, v_other, weight_array)
                        
                        # Spaltenname: z.B. "S_avg_average" oder "S_topN_weighted_base"
                        col_name = f"S_{metric_name.replace('proximity_', '')}_{weight_name}"
                        row[col_name] = sim
                        
                    except (ValueError, ZeroDivisionError) as e:
                        col_name = f"S_{metric_name.replace('proximity_', '')}_{weight_name}"
                        row[col_name] = np.nan
                        if verbose:
                            print(f"  Warnung: {metric_name} mit {weight_name} fehlgeschlagen für {ref_id}-{other_id}: {e}")
            
            rows.append(row)
    
    # DataFrame erstellen mit sortierter Spaltenreihenfolge
    df = pd.DataFrame(rows)
    
    # Spalten sortieren: IDs zuerst, dann alphabetisch
    id_cols = ["Referenzkunde_ID", "Kunde_ID"]
    metric_cols = sorted([c for c in df.columns if c not in id_cols])
    df = df[id_cols + metric_cols]
    
    return df

# In benchmark_of_semantic_operators.py ergänzen

def run_experiment2_weighting_analysis(
    model_list,
    df_with_keys: pd.DataFrame,
    weight_vectors: dict,
    metrics_to_use: List[str] = ['proximity_avg', 'proximity_topn_avg'],
    topn_n: int = 3,
    subset_k: int = 3,
    top_n: int = 200,
    base_dir: str = ".",
    verbose: bool = False
) -> dict:
    """
    Hauptfunktion für Experiment 2: Averaging vs. Weighting Analyse.
    
    Läuft ähnlich wie run_full_vector_test(), aber mit Fokus auf Gewichtungen.
    
    Parameters:
    -----------
    model_list : list of tuples
        Liste von (model, model_info) Tupeln
    df_with_keys : pd.DataFrame
        DataFrame mit allen Kunden
    weight_vectors : dict
        Dictionary mit Gewichtungsvektoren
    metrics_to_use : List[str]
        Welche Basis-Metriken verwendet werden sollen
    topn_n, subset_k : int
        Parameter für entsprechende Metriken
    top_n : int
        Anzahl für Top/Bottom Rankings (default: 200)
    base_dir : str
        Verzeichnis für Zwischenspeicherung
    verbose : bool
        Fortschrittsanzeige
    
    Returns:
    --------
    dict : Alle Ergebnisse strukturiert
    """
    results = {}
    
    # Referenzkunden auswählen (WIEDERVERWENDET!)
    selected_customers, cluster_centers, selected_cluster_info, kmeans = \
        get_diverse_customers_with_clusters(df_with_keys, k=5, verbose=verbose)
    
    if verbose:
        print("\n" + "="*70)
        print("EXPERIMENT 2: AVERAGING vs. WEIGHTING ANALYSE")
        print("="*70)
        print(f"\nGewichtungsvektoren: {list(weight_vectors.keys())}")
        print(f"Metriken: {metrics_to_use}")
        print("\nAusgewählte diverse Kunden für Tests:")
        print(selected_customers[['cluster_group', 'cluster_size']].to_string(index=False))
    
    for model_idx, (model, model_info) in enumerate(model_list):
        if verbose:
            print(f"\n{'='*60}")
            print(f"MODEL {model_idx}: {model_info}")
            print(f"{'='*60}\n")
        
        # PHASE 1: Gewichtete Ähnlichkeitsberechnung
        if verbose:
            print("PHASE 1: Gewichtete Ähnlichkeitsberechnung")
        
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
            # Neu berechnen
            phase1_df = compute_phase1_weighted_similarity_df(
                model=model,
                df_with_keys=df_with_keys,
                selected_customers=selected_customers,
                weight_vectors=weight_vectors,
                metrics_to_use=metrics_to_use,
                topn_n=topn_n,
                subset_k=subset_k,
                verbose=verbose
            )
            
            # Speichern
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
        
        # PHASE 2a: Rankings (KOMPLETT WIEDERVERWENDET!)
        if verbose:
            print("\nPHASE 2a: Rankings")
        
        rankings_data = save_or_load_phase_results(
            phase_name="exp2_phase2_rankings",
            phase_data=None,
            base_dir=base_dir,
            model_idx=model_idx,
            model_info=model_info,
            selected_customers=selected_customers,
            verbose=verbose
        )
        
        if rankings_data is None:
            rankings = create_rankings_for_phase2(phase1_df, top_n=top_n, verbose=verbose)
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
            rankings = rankings_data
        
        results[f"exp2_phase2_rankings_model{model_idx}"] = rankings
        
        # PHASE 2b: Overlaps (WIEDERVERWENDET!)
        if verbose:
            print("\nPHASE 2b: Overlaps zwischen Metriken")
        
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
            overlaps = compute_overlap_coefficients(rankings, verbose=verbose)
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
        
        # PHASE 3: Statistische Charakterisierung (KOMPLETT WIEDERVERWENDET!)
        if verbose:
            print("\nPHASE 3: Statistische Charakterisierung")
        
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
        
        # PHASE 4: Separation Scores (KOMPLETT WIEDERVERWENDET!)
        if verbose:
            print("\nPHASE 4: Diskriminierungsfähigkeit (Separation Score)")
        
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
    
    return results

# Neue Auswertungsfunktionen für Experiment 2

def compute_exp2_separation_matrix(phase4_results: dict, 
                                   weight_names: List[str],
                                   base_metrics: List[str] = ['avg', 'topn_avg']) -> pd.DataFrame:
    """
    Erstellt Separation Power Matrix für Darstellung 1.
    
    Parameters:
    -----------
    phase4_results : dict
        Output von compute_phase4_separation_scores
    weight_names : List[str]
        Liste der Gewichtungsnamen (z.B. ['average', 'weighted_base', ...])
    base_metrics : List[str]
        Basis-Metriknamen ohne 'proximity_' Präfix (z.B. ['avg', 'topn_avg'])
    
    Returns:
    --------
    pd.DataFrame : Matrix mit Zeilen=Gewichtungen, Spalten=Metriken
    """
    summary_df = phase4_results['summary_df']
    
    # Matrix vorbereiten
    matrix_data = []
    
    for weight_name in weight_names:
        row_data = {'Gewichtung': weight_name}
        
        for base_metric in base_metrics:
            # Suche in summary_df nach Spalte, die diesem Pattern entspricht
            # z.B. "S_avg_average" für base_metric='avg' und weight_name='average'
            col_pattern = f"S_{base_metric}_{weight_name}"
            
            # Finde die entsprechende Zeile in summary_df
            matching_rows = summary_df[summary_df['Metrik'] == col_pattern]
            
            if len(matching_rows) > 0:
                separation_score = matching_rows.iloc[0]['separation_mean']
                row_data[f"S_{base_metric}"] = separation_score
            else:
                row_data[f"S_{base_metric}"] = np.nan
        
        # Durchschnitt über alle Metriken
        metric_values = [v for k, v in row_data.items() if k != 'Gewichtung']
        row_data['Durchschnitt'] = np.nanmean(metric_values) if metric_values else np.nan
        
        matrix_data.append(row_data)
    
    matrix_df = pd.DataFrame(matrix_data)
    
    return matrix_df


def compute_exp2_overlap_curves(
    rankings_dict: dict,
    weight_names: List[str],
    baseline_weight: str = 'average',
    metric_to_analyze: str = 'S_avg_average',  # Eine spezifische Metrik auswählen
    max_k: int = 200,
    verbose: bool = False
) -> dict:
    """
    Berechnet Overlap-Kurven für Darstellung 2.
    
    Vergleicht Rankings von baseline_weight mit allen anderen Gewichtungen
    für eine ausgewählte Basis-Metrik.
    
    Parameters:
    -----------
    rankings_dict : dict
        Output von create_rankings_for_phase2
    weight_names : List[str]
        Liste aller Gewichtungsnamen
    baseline_weight : str
        Baseline-Gewichtung (typisch 'average')
    metric_to_analyze : str
        Welche Metrik analysiert werden soll (muss Gewichtung enthalten!)
        Z.B. 'S_avg_average', 'S_avg_weighted_base', etc.
    max_k : int
        Maximale K für Top-K Analyse
    verbose : bool
        Fortschrittsanzeige
    
    Returns:
    --------
    dict : {
        'curves': {
            weight_name: [overlap_at_k1, overlap_at_k2, ..., overlap_at_k200]
        },
        'optimal_curve': [1, 2, 3, ..., 200],  # Diagonale
        'metric_analyzed': str
    }
    """
    if verbose:
        print(f"Berechne Overlap-Kurven für Metrik: {metric_to_analyze}")
    
    # Extrahiere Basis-Metrikname (z.B. 'avg' aus 'S_avg_average')
    parts = metric_to_analyze.split('_')
    if len(parts) < 3:
        raise ValueError(f"Ungültiger Metrikname: {metric_to_analyze}")
    
    base_metric = '_'.join(parts[:-1])  # z.B. "S_avg"
    
    curves = {}
    reference_ids = list(rankings_dict.keys())
    
    # Für jede Gewichtung (außer Baseline)
    for weight_name in weight_names:
        if weight_name == baseline_weight:
            continue
        
        if verbose:
            print(f"  Verarbeite Gewichtung: {weight_name}")
        
        overlap_curve = []
        
        # Für jedes K von 1 bis max_k
        for k in range(1, max_k + 1):
            overlaps_at_k = []
            
            # Für jeden Referenzkunden
            for ref_id in reference_ids:
                # Baseline-Ranking
                baseline_metric_name = f"{base_metric}_{baseline_weight}"
                if baseline_metric_name not in rankings_dict[ref_id]['top']:
                    continue
                
                # Weighted-Ranking
                weighted_metric_name = f"{base_metric}_{weight_name}"
                if weighted_metric_name not in rankings_dict[ref_id]['top']:
                    continue
                
                # Top-K IDs holen
                topk_baseline = set(rankings_dict[ref_id]['top'][baseline_metric_name][:k])
                topk_weighted = set(rankings_dict[ref_id]['top'][weighted_metric_name][:k])
                
                # Overlap berechnen
                overlap = len(topk_baseline.intersection(topk_weighted))
                overlaps_at_k.append(overlap)
            
            # Durchschnitt über alle Referenzkunden
            avg_overlap_at_k = np.mean(overlaps_at_k) if overlaps_at_k else 0
            overlap_curve.append(avg_overlap_at_k)
        
        curves[weight_name] = overlap_curve
    
    # Optimalkurve (perfekte Übereinstimmung)
    optimal_curve = list(range(1, max_k + 1))
    
    return {
        'curves': curves,
        'optimal_curve': optimal_curve,
        'metric_analyzed': metric_to_analyze,
        'baseline_weight': baseline_weight,
        'max_k': max_k
    }


def compute_exp2_weight_impact_summary(
    overlap_curves_result: dict,
    threshold_k: int = 50
) -> pd.DataFrame:
    """
    Berechnet zusammenfassende Statistiken aus den Overlap-Kurven.
    
    Parameters:
    -----------
    overlap_curves_result : dict
        Output von compute_exp2_overlap_curves
    threshold_k : int
        Bei welchem Top-K die spezifische Abweichung gemessen wird
        Returns:
    --------
    pd.DataFrame : Zusammenfassung mit Impact-Metriken pro Gewichtung
    """
    curves = overlap_curves_result['curves']
    optimal_curve = overlap_curves_result['optimal_curve']

    summary_rows = []

    for weight_name, curve in curves.items():
        # Durchschnittliche Abweichung von Optimalkurve
        deviations = [optimal_curve[i] - curve[i] for i in range(len(curve))]
        avg_deviation = np.mean(deviations)
        max_deviation = np.max(deviations)
        
        # Abweichung bei spezifischem K
        deviation_at_k = optimal_curve[threshold_k - 1] - curve[threshold_k - 1]
        
        # Bereich, wo Abweichung > 10% des optimalen Werts
        significant_deviations = [i+1 for i, d in enumerate(deviations) 
                                if d > 0.1 * optimal_curve[i]]
        first_significant_k = significant_deviations[0] if significant_deviations else None
        
        summary_rows.append({
            'Gewichtung': weight_name,
            'Durchschn_Abweichung': avg_deviation,
            'Max_Abweichung': max_deviation,
            f'Abweichung_bei_K{threshold_k}': deviation_at_k,
            'Erste_signifikante_Abweichung_K': first_significant_k,
            'Impact_Score': avg_deviation / optimal_curve[-1]  # Normalisierter Impact
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values('Impact_Score', ascending=False)

    return summary_df