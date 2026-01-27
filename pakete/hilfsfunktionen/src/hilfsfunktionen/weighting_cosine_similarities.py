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
from typing import List, Tuple, Iterable, Dict, Optional


def compute_phase1_weighted_similarity_df(
    model,
    df_with_keys: pd.DataFrame,
    selected_customers: pd.DataFrame,
    weight_vectors: dict,
    metrics_to_use: List[str] = ['proximity_avg'],
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
    churn_col: str = "churn",
    run_churn_propensity: bool = True,
    churn_include_churned: bool = False,
    churn_use_zscores: bool = True,
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

        # PHASE 3B (Szenario B): Churn-Propensity (CONTRASTIVE)
        if run_churn_propensity:
            if verbose:
                print("\nPHASE 3B: Churn-Propensity (CONTRASTIVE)")

            churn_phase_name = "exp2_phase3b_churn_propensity_contrastive"

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

            # Robuste Top-10 Kandidaten (Konsens über Gewichtungen)
            weight_names = list(weight_vectors.keys())
            top10_candidates = select_top_churn_candidates(
                churn_scores_df=churn_scores_df,
                weight_names=weight_names,
                k_final=10,
                pool_k=20,
                min_appearances=max(2, min(3, len(weight_names))),
                use_zscores=churn_use_zscores
            )

            results[f"exp2_phase3b_churn_scores_model{model_idx}"] = churn_scores_df
            results[f"exp2_phase3b_top10_churn_candidates_model{model_idx}"] = top10_candidates


        
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


# ---------------------------------------------------------------------
# Experiment 2 – Phase 3 (Szenario B): Churn-Propensity (Similarity-to-Churn-Centroid)
# ---------------------------------------------------------------------

def _safe_weight_array(weight_vec: Iterable[float], dim: int, weight_name: str) -> np.ndarray:
    """Konvertiert weight_vec zu float-array und validiert Dimension."""
    w = np.asarray(list(weight_vec), dtype=float)
    if w.ndim != 1:
        raise ValueError(f"Gewichtungsvektor '{weight_name}' ist nicht 1-dimensional.")
    if len(w) != dim:
        raise ValueError(
            f"Dimension-Mismatch für Gewichtung '{weight_name}': "
            f"len(w)={len(w)} vs. embedding_dim={dim}. "
            "Bitte sicherstellen, dass die Reihenfolge/DIMs zum Word2Vec-Vektor passen."
        )
    return w


def _cosine_sim(u: np.ndarray, v: np.ndarray) -> float:
    """Numerisch robuste Kosinus-Ähnlichkeit; gibt np.nan bei Nullnorm zurück."""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    du = np.linalg.norm(u)
    dv = np.linalg.norm(v)
    if du == 0.0 or dv == 0.0:
        return np.nan
    return float(np.dot(u, v) / (du * dv))


def _customer_centroids_from_vectorlists(all_vectors: List[List[np.ndarray]]) -> np.ndarray:
    """
    Erzeugt pro Kunde einen Centroid (Mean) über die Key-Vektoren.
    Rückgabe: (n_customers, embedding_dim)
    """
    centroids = []
    for vec_list in all_vectors:
        if vec_list is None or len(vec_list) == 0:
            centroids.append(None)
            continue
        mat = np.vstack(vec_list)  # (n_keys, dim)
        centroids.append(mat.mean(axis=0))
    # Dimensionsbestimmung über den ersten validen Centroid
    first = next((c for c in centroids if c is not None), None)
    if first is None:
        raise ValueError("Keine gültigen Vektoren gefunden – können keine Centroids bilden.")
    dim = int(len(first))
    out = np.zeros((len(centroids), dim), dtype=float)
    for i, c in enumerate(centroids):
        out[i, :] = c if c is not None else 0.0
    return out


def compute_churn_centroids_weighted(
    model,
    df_with_keys: pd.DataFrame,
    weight_vectors: Dict[str, Iterable[float]],
    churn_col: str = "churn",
    verbose: bool = False
) -> Dict[str, np.ndarray]:
    """
    Berechnet pro Gewichtung einen Churn-Centroid (Centroid über alle churned Kunden).
    Vorgehen:
      1) Pro Kunde: Centroid über Key-Vektoren.
      2) Gewichten (elementweise Multiplikation).
      3) Über alle churned Kunden mitteln.

    Returns:
      dict: weight_name -> churn_centroid (np.ndarray, dim)
    """
    if churn_col not in df_with_keys.columns:
        raise ValueError(f"Spalte '{churn_col}' fehlt in df_with_keys (benötigt für Churn-Propensity).")

    churn_mask = (df_with_keys[churn_col].astype(int).to_numpy() == 1)
    if churn_mask.sum() == 0:
        raise ValueError("Keine churned Kunden (churn=1) im Datensatz – Churn-Centroid nicht berechenbar.")

    all_vectors = _compute_all_customer_vectors(model, df_with_keys)
    centroids = _customer_centroids_from_vectorlists(all_vectors)
    dim = centroids.shape[1]

    churn_centroids = {}
    churn_centroid_base = centroids[churn_mask, :]  # (n_churn, dim)

    for weight_name, weight_vec in weight_vectors.items():
        w = _safe_weight_array(weight_vec, dim=dim, weight_name=weight_name)

        # Gewichtung im Vektorraum
        weighted_churn = churn_centroid_base * w  # Broadcasting

        # Mittelwert (Centroid) über churned Kunden
        churn_centroid_w = weighted_churn.mean(axis=0)

        churn_centroids[weight_name] = churn_centroid_w

        if verbose:
            print(f"[ChurnCentroid] {weight_name}: dim={dim}, n_churn={int(churn_mask.sum())}")

    return churn_centroids


def compute_churn_propensity_scores(
    model,
    df_with_keys: pd.DataFrame,
    weight_vectors: Dict[str, Iterable[float]],
    churn_col: str = "churn",
    include_churned: bool = False,
    zscore: bool = True,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Phase 3 (Szenario B): Churn-Propensity je Kunde als Ähnlichkeit zum Churn-Centroid.

    - Für jeden Non-Churn-Kunden wird die Ähnlichkeit zum (gewichteten) Churn-Centroid berechnet.
    - Optional können churned Kunden mit ausgegeben werden (include_churned=True).
    - Optional werden Z-Scores pro Gewichtung berechnet (zscore=True), um Weightings vergleichbar zu machen.

    Returns:
      DataFrame mit Spalten:
        Kunde_ID, churn, ChurnProp_<weight>, ChurnPropZ_<weight>, Rank_<weight>
    """
    df_with_keys_processed = df_with_keys.copy()
    df_with_keys_processed['churn'] = (df_with_keys_processed['churn'] == 'Churn_Yes').astype(int)

    DEBUG = True
    def dprint(*args, **kwargs):
        if DEBUG and verbose:
            print(*args, **kwargs)
    
    dprint("[DEBUG] churn value counts (raw):", df_with_keys_processed["churn"].value_counts(dropna=False).to_dict())
    dprint("[DEBUG] churn dtype:", df_with_keys_processed["churn"].dtype)
    dprint("[DEBUG] n_customers:", len(df_with_keys_processed))
    dprint("[DEBUG] n_churned:", int((df_with_keys_processed["churn"] == 1).sum()))

    churn_mask = (df_with_keys_processed[churn_col].astype(int).to_numpy() == 1)
    active_mask = ~churn_mask
    
    all_vectors = _compute_all_customer_vectors(model, df_with_keys)

    # DEBUG 2: Struktur von all_vectors
    dprint("[DEBUG] all_vectors type:", type(all_vectors))
    dprint("[DEBUG] n_customers in all_vectors:", len(all_vectors))
    
    lens = [len(vs) for vs in all_vectors[:50]]
    dprint("[DEBUG] first 50 feature-counts:", lens)
    if lens:
        dprint("[DEBUG] feature-count min/max (first 50):", min(lens), max(lens))
    
    # DEBUG 3: Vektor-Dimensionen und Konsistenz
    def vec_dim(x):
        try:
            return int(np.asarray(x).shape[0])
        except Exception:
            return None
    
    # Prüfen an einem Kunden
    i = 0
    if all_vectors and len(all_vectors[i]) > 0:
        dims_i = [vec_dim(v) for v in all_vectors[i][:10]]
        dprint(f"[DEBUG] customer {i} first 10 vector dims:", dims_i)
    
    # Prüfen über mehrere Kunden und Features
    sample_dims = []
    for ci in range(min(10, len(all_vectors))):
        if len(all_vectors[ci]) > 0:
            for fi in range(min(5, len(all_vectors[ci]))):
                sample_dims.append(vec_dim(all_vectors[ci][fi]))
    
    if sample_dims:
        dprint("[DEBUG] sample dims (10 customers x 5 feats):", sample_dims)
        dprint("[DEBUG] unique dims:", sorted(set(sample_dims)))
    
    n_features = len(all_vectors[0]) if all_vectors[0] else 0
    
    base_df = pd.DataFrame({
        "Kunde_ID": np.arange(len(df_with_keys_processed), dtype=int),
        churn_col: df_with_keys_processed[churn_col].astype(int).to_numpy()
    })
    
    # DEBUG 4: Gewichtungsvektoren – Anzahl, Längen, Verteilung (VOR der Schleife!)
    dprint("[DEBUG] expected n_features:", n_features)
    dprint("[DEBUG] number of weight vectors:", len(weight_vectors))
    
    for weight_name, weight_vec in weight_vectors.items():
        w = np.asarray(list(weight_vec))
        dprint(f"[DEBUG] w['{weight_name}'] len={len(w)} sum={float(w.sum()):.6f} min={float(w.min()):.6f} max={float(w.max()):.6f} zeros={(w==0).sum()}")
    
    for weight_name, weight_vec in weight_vectors.items():
        w = np.asarray(list(weight_vec), dtype=float)
        
        if len(w) != n_features:
            raise ValueError(f"Gewichtung '{weight_name}' hat falsche Dimension! {len(w)} vs {n_features}")
        
        # --- CHURN-CENTROID: Gewichteter Zentroid über alle churned ---
        churn_weighted_centroids = []
        for cust_idx in np.where(churn_mask)[0]:
            if all_vectors[cust_idx]:
                from .cosine_funktions import weighted_centroid
                wc = weighted_centroid(all_vectors[cust_idx], w)
                churn_weighted_centroids.append(wc)
        
        # DEBUG 6: Churn-Centroid-Logik
        dprint(f"[DEBUG] churned_centroids count for {weight_name}:", len(churn_weighted_centroids))
        
        if not churn_weighted_centroids:
            raise ValueError(f"Keine gewichteten Centroids für churned Kunden mit Gewichtung '{weight_name}' berechnet!")
        
        # Churn-Centroid = Mittelwert der gewichteten Zentroiden
        churn_centroid_w = np.mean(np.vstack(churn_weighted_centroids), axis=0)
        dprint(f"[DEBUG] churn_centroid dim for {weight_name}:", churn_centroid_w.shape)
        
        # --- FÜR JEDEN KUNDEN: Ähnlichkeit zum Churn-Centroid ---
        scores = []
        for cust_idx in range(len(df_with_keys)):
            if all_vectors[cust_idx]:
                from .cosine_funktions import weighted_centroid
                cust_centroid_w = weighted_centroid(all_vectors[cust_idx], w)
                sim = _cosine_sim(cust_centroid_w, churn_centroid_w)
                scores.append(sim)
            else:
                scores.append(np.nan)
        
        scores = np.array(scores)
        
        if not include_churned:
            scores = np.where(active_mask, scores, np.nan)
        
        # DEBUG 7: Score-Verteilung (NUR für die ERSTE Gewichtung)
        if weight_name == list(weight_vectors.keys())[0] and scores.size > 0:
            valid_scores = scores[~np.isnan(scores)]
            if len(valid_scores) > 0:
                dprint(f"[DEBUG] {weight_name} sims stats: min/mean/median/max =", 
                      float(valid_scores.min()), float(valid_scores.mean()), 
                      float(np.median(valid_scores)), float(valid_scores.max()))
                if len(valid_scores) >= 10:
                    dprint(f"[DEBUG] {weight_name} sims quantiles:", 
                          np.quantile(valid_scores, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]).tolist())
            
            # Top-10 Indizes für erste Gewichtung
            valid_indices = np.where(~np.isnan(scores))[0]
            if len(valid_indices) > 0:
                valid_scores = scores[valid_indices]
                top_idx = valid_indices[np.argsort(-valid_scores)[:10]]
                dprint(f"[DEBUG] {weight_name} top10 idx:", top_idx.tolist())
                dprint(f"[DEBUG] {weight_name} top10 sims:", scores[top_idx].tolist())
                dprint(f"[DEBUG] {weight_name} top10 churn labels:", df_with_keys_processed.iloc[top_idx]["churn"].tolist())
        
        score_col = f"ChurnProp_{weight_name}"
        base_df[score_col] = scores
        
        rank_col = f"Rank_{weight_name}"
        base_df[rank_col] = base_df[score_col].rank(ascending=False, method="min")
        
        if zscore:
            z_col = f"ChurnPropZ_{weight_name}"
            valid = base_df[score_col].dropna()
            mu = valid.mean() if len(valid) > 0 else np.nan
            sd = valid.std(ddof=0) if len(valid) > 0 else np.nan
            if sd == 0 or np.isnan(sd):
                base_df[z_col] = np.nan
            else:
                base_df[z_col] = (base_df[score_col] - mu) / sd

    # DEBUG 8: Feature-wise Vergleich (optional)
    if verbose and len(all_vectors) > 0:
        debug_featurewise_comparison(model, df_with_keys_processed, weight_vectors, all_vectors, churn_mask)
    
    return base_df


def select_top_churn_candidates(
    churn_scores_df: pd.DataFrame,
    weight_names: List[str],
    k_final: int = 10,
    pool_k: int = 20,
    min_appearances: int = 3,
    use_zscores: bool = True
) -> pd.DataFrame:
    """
    Liefert eine robuste Top-k Liste, die nicht nur von einer Gewichtung abhängt.

    Vorgehen:
      1) Kandidatenpool = Kunden, die in mindestens min_appearances Gewichtungen in Top-pool_k sind.
      2) Sortierung innerhalb des Pools nach Mittelwert der Z-Scores (oder Rohscores).

    Returns:
      DataFrame mit Kunde_ID, appearances, mean_score, min_rank, plus per-weight rank/score.
    """
    df = churn_scores_df.copy()

    # Nur Non-Churn (churn=0) sinnvoll, falls Scores NaN für churned
    # Wir filtern über vorhandene Score-Spalten (NaN -> fällt raus)
    score_cols = []
    rank_cols = []
    for w in weight_names:
        score_cols.append(f"ChurnPropZ_{w}" if use_zscores and f"ChurnPropZ_{w}" in df.columns else f"ChurnProp_{w}")
        rank_cols.append(f"Rank_{w}")

    # appearances: wie oft Rank <= pool_k
    appearances = np.zeros(len(df), dtype=int)
    for rc in rank_cols:
        if rc in df.columns:
            appearances += (df[rc].to_numpy() <= pool_k).astype(int)

    df["appearances"] = appearances

    # Pool-Filter
    pool_df = df[df["appearances"] >= int(min_appearances)].copy()

    # Aggregatscores
    pool_df["mean_score"] = pool_df[score_cols].mean(axis=1, skipna=True)
    pool_df["min_rank"] = pool_df[rank_cols].min(axis=1, skipna=True)

    # Sortierung: erst appearances, dann mean_score, dann min_rank
    pool_df = pool_df.sort_values(["appearances", "mean_score", "min_rank"], ascending=[False, False, True])

    # Falls Pool zu klein ist, fallback: beste mean_score über alle
    if len(pool_df) < k_final:
        fallback = df.copy()
        fallback["mean_score"] = fallback[score_cols].mean(axis=1, skipna=True)
        fallback["min_rank"] = fallback[rank_cols].min(axis=1, skipna=True)
        fallback = fallback.sort_values(["mean_score", "min_rank"], ascending=[False, True])
        pool_df = fallback

    keep_cols = ["Kunde_ID", "appearances", "mean_score", "min_rank"]
    # per-weight columns dazu
    for w in weight_names:
        for c in (f"ChurnProp_{w}", f"ChurnPropZ_{w}", f"Rank_{w}"):
            if c in pool_df.columns and c not in keep_cols:
                keep_cols.append(c)

    return pool_df[keep_cols].head(int(k_final))


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

def profile_top_churn_candidates(
    df_customers: pd.DataFrame,
    candidate_ids: Iterable,
    *,
    id_col: Optional[str] = None,
    churn_col: str = "churned",
    feature_cols: Optional[List[str]] = None,
    top_numeric_features: int = 10,
) -> Dict[str, pd.DataFrame]:
    """Profile top churn-propensity candidates against (a) overall population and (b) churned customers.

    Returns a dict with:
      - 'overall_statistics': Global statistics including baseline churn rate
      - 'global_risk_features': Top 20 riskiest feature values globally
      - 'candidate_overview': one row per candidate with id and churn label
      - 'numeric_feature_profile': long table (candidate, feature) with candidate value, overall/churned means
      - 'numeric_top_drivers': per candidate the top numeric features by |z|
      - 'categorical_feature_profile': long table with conditional churn rates and comparison to baseline
    """

    if id_col is None:
        for c in ["Customer_ID", "Kunde_ID", "customer_id", "kunde_id", "RowNumber", "ID"]:
            if c in df_customers.columns:
                id_col = c
                break
        if id_col is None:
            raise ValueError(
                "Could not infer id_col. Please pass id_col explicitly (e.g., 'Customer_ID' or 'Kunde_ID')."
            )

    if churn_col not in df_customers.columns:
        raise ValueError(f"churn_col='{churn_col}' not found in df_customers columns.")

    cand_ids = list(candidate_ids)
    df_cand = df_customers[df_customers[id_col].isin(cand_ids)].copy()
    if df_cand.empty:
        raise ValueError("No candidate rows found. Check id_col and candidate_ids.")

    # ========== KORRIGIERTE CHURN-KONVERTIERUNG ==========
    # Erstelle Kopien für die Analyse
    df_all = df_customers.copy()
    
    # Prüfe die tatsächlichen Werte in der churn-Spalte
    unique_churn_values = df_all[churn_col].unique()
    print(f"Unique churn values found: {unique_churn_values}")
    
    # Konvertiere churn-Spalte zu numerisch 0/1
    if df_all[churn_col].dtype.name == 'category' or df_all[churn_col].dtype == 'object':
        # Automatische Erkennung von Churn-Indikatoren
        churn_positive_indicators = ['Churn_Yes', 'Yes', 'yes', '1', 'True', 'true', 'churn', 'Churn']
        churn_negative_indicators = ['Churn_No', 'No', 'no', '0', 'False', 'false', 'no churn', 'No Churn']
        
        # Finde den passenden Indikator
        churn_positive_found = None
        for indicator in churn_positive_indicators:
            if any(indicator.lower() in str(val).lower() for val in unique_churn_values):
                churn_positive_found = indicator
                break
        
        if churn_positive_found:
            # Erstelle numerische Spalte basierend auf String-Matching
            df_all['churn_numeric'] = df_all[churn_col].apply(
                lambda x: 1 if churn_positive_found.lower() in str(x).lower() else 0
            )
            print(f"Using '{churn_positive_found}' as churn=1 indicator")
        else:
            # Fallback: Wenn 'Yes'/'No' nicht gefunden, nimm den ersten Wert als 1
            print(f"Warning: No standard churn indicator found. Using first value as churn=1")
            first_val = str(unique_churn_values[0])
            df_all['churn_numeric'] = df_all[churn_col].apply(
                lambda x: 1 if str(x) == first_val else 0
            )
        
        # Für Kandidaten-Datenframe ebenfalls konvertieren
        df_cand = df_cand.copy()
        if 'churn_numeric' not in df_cand.columns:
            if churn_positive_found:
                df_cand['churn_numeric'] = df_cand[churn_col].apply(
                    lambda x: 1 if churn_positive_found.lower() in str(x).lower() else 0
                )
            else:
                df_cand['churn_numeric'] = df_cand[churn_col].apply(
                    lambda x: 1 if str(x) == str(unique_churn_values[0]) else 0
                )
        
        # Verwende die numerische Spalte für die Analyse
        churn_col_numeric = 'churn_numeric'
    else:
        # Bereits numerisch
        churn_col_numeric = churn_col
        try:
            df_all[churn_col_numeric] = df_all[churn_col].astype(int)
            df_cand[churn_col_numeric] = df_cand[churn_col].astype(int)
        except Exception as e:
            raise ValueError(f"churn_col='{churn_col}' must be numeric (0/1) or castable to int. Error: {e}") from e
    
    # ========== AB HIER MIT NUMERISCHER CHURN-SPALTE ARBEITEN ==========
    
    df_churned = df_all[df_all[churn_col_numeric] == 1]
    if df_churned.empty:
        raise ValueError(f"No churned customers found ({churn_col_numeric}=1). Cannot compute churned baseline profiles.")

    # Berechne globale Statistiken
    overall_churn_rate = df_all[churn_col_numeric].mean()
    overall_churn_count = int(df_all[churn_col_numeric].sum())
    overall_total = len(df_all)
    
    # Churn-Rate der Kandidaten
    candidate_churn_rate = df_cand[churn_col_numeric].mean() if not df_cand.empty else np.nan
    print(f"Overall churn rate: {overall_churn_rate:.2%}")
    print(f"Candidate churn rate: {candidate_churn_rate:.2%}")

    # Default feature columns (exclude the temporary churn_numeric column)
    if feature_cols is None:
        exclude = {id_col, churn_col, 'churn_numeric'}
        for maybe in ["keys", "token_keys", "embedding", "vectors", "cluster", "Cluster", "reference_id"]:
            if maybe in df_all.columns:
                exclude.add(maybe)
        feature_cols = [c for c in df_all.columns if c not in exclude]

    # Erkennung numerischer/kategorialer Spalten
    numeric_cols = []
    categorical_cols = []
    
    for c in feature_cols:
        dtype = df_all[c].dtype
        if pd.api.types.is_numeric_dtype(dtype):
            numeric_cols.append(c)
        else:
            categorical_cols.append(c)
    
    print(f"Found {len(numeric_cols)} numeric features and {len(categorical_cols)} categorical features")

    # ========== Top Risikomerkmale global ==========
    global_risk_features = []
    for f in categorical_cols:
        unique_values = df_all[f].dropna().unique() if not df_all[f].isna().all() else []
        
        for val in unique_values[:50]:  # Limitiere für Performance
            mask = (df_all[f] == val)
            if mask.sum() > 0:
                count_total = int(mask.sum())
                count_churned = int((mask & (df_all[churn_col_numeric] == 1)).sum())
                
                if count_total >= 10:  # Mindestanzahl für statistische Aussage
                    churn_rate = count_churned / count_total
                    global_risk_features.append({
                        "feature": f,
                        "value": val,
                        "total_count": count_total,
                        "churned_count": count_churned,
                        "churn_rate": churn_rate,
                        "churn_rate_ratio": churn_rate / overall_churn_rate if overall_churn_rate > 0 else np.nan,
                        "churn_rate_delta": churn_rate - overall_churn_rate,
                    })
    
    global_risk_df = pd.DataFrame(global_risk_features)
    if not global_risk_df.empty:
        global_top_risks = global_risk_df.sort_values(
            "churn_rate", ascending=False
        ).head(20).reset_index(drop=True)
    else:
        global_top_risks = pd.DataFrame()

    # Baselines for numeric features
    num_overall_mean = df_all[numeric_cols].mean(numeric_only=True)
    num_overall_std = df_all[numeric_cols].std(ddof=0, numeric_only=True).replace(0, np.nan)
    num_churn_mean = df_churned[numeric_cols].mean(numeric_only=True)

    # Numeric profiling
    numeric_rows = []
    for _, row in df_cand.iterrows():
        cid = row[id_col]
        for f in numeric_cols:
            v = row[f]
            om = num_overall_mean.get(f, np.nan)
            osd = num_overall_std.get(f, np.nan)
            cm = num_churn_mean.get(f, np.nan)
            z = (v - om) / osd if pd.notna(v) and pd.notna(om) and pd.notna(osd) else np.nan
            numeric_rows.append(
                {
                    id_col: cid,
                    "feature": f,
                    "candidate_value": v,
                    "overall_mean": om,
                    "churned_mean": cm,
                    "z_vs_overall": z,
                    "delta_vs_overall": v - om if pd.notna(v) and pd.notna(om) else np.nan,
                    "delta_vs_churned": v - cm if pd.notna(v) and pd.notna(cm) else np.nan,
                }
            )
    numeric_profile = pd.DataFrame(numeric_rows)

    # Top drivers per candidate (largest |z|)
    drivers = []
    if not numeric_profile.empty:
        for cid, g in numeric_profile.groupby(id_col):
            g2 = g.dropna(subset=["z_vs_overall"]).copy()
            g2["abs_z"] = g2["z_vs_overall"].abs()
            topg = g2.sort_values("abs_z", ascending=False).head(top_numeric_features)
            for _, r in topg.iterrows():
                drivers.append(
                    {
                        id_col: cid,
                        "feature": r["feature"],
                        "z_vs_overall": r["z_vs_overall"],
                        "candidate_value": r["candidate_value"],
                        "overall_mean": r["overall_mean"],
                        "churned_mean": r["churned_mean"],
                    }
                )
    numeric_top_drivers = pd.DataFrame(drivers)

    # Categorical profiling MIT Baseline-Vergleich
    cat_rows = []
    for _, row in df_cand.iterrows():
        cid = row[id_col]
        for f in categorical_cols:
            val = row[f]
            
            if pd.isna(val):
                mask_val = df_all[f].isna()
            else:
                mask_val = (df_all[f] == val)

            overall_count = int(mask_val.sum())
            overall_freq = overall_count / len(df_all) if len(df_all) else np.nan

            churned_count = int((mask_val & (df_all[churn_col_numeric] == 1)).sum())
            churned_freq_within_churned = churned_count / len(df_churned) if len(df_churned) else np.nan

            churn_rate_for_value = churned_count / overall_count if overall_count else np.nan
            
            # Vergleich mit Baseline
            churn_rate_ratio = churn_rate_for_value / overall_churn_rate if overall_churn_rate > 0 else np.nan
            churn_rate_delta = churn_rate_for_value - overall_churn_rate
            
            # Risikoklassifikation
            risk_level = "neutral"
            if churn_rate_ratio > 2.0:
                risk_level = "very_high"
            elif churn_rate_ratio > 1.5:
                risk_level = "high"
            elif churn_rate_ratio > 1.2:
                risk_level = "elevated"
            elif churn_rate_ratio < 0.8:
                risk_level = "low_risk"

            cat_rows.append(
                {
                    id_col: cid,
                    "feature": f,
                    "candidate_value": val,
                    "overall_freq": overall_freq,
                    "overall_count": overall_count,
                    "churned_freq": churned_freq_within_churned,
                    "churned_count": churned_count,
                    "conditional_churn_rate": churn_rate_for_value,
                    
                    # NEUE SPALTEN:
                    "overall_churn_rate_baseline": overall_churn_rate,
                    "churn_rate_ratio_vs_baseline": churn_rate_ratio,
                    "churn_rate_delta_vs_baseline": churn_rate_delta,
                    "risk_level": risk_level,
                }
            )
    categorical_profile = pd.DataFrame(cat_rows)

    candidate_overview = df_cand[[id_col, churn_col, churn_col_numeric]].copy()
    candidate_overview = candidate_overview.rename(columns={
        churn_col: "is_churned_original",
        churn_col_numeric: "is_churned_numeric"
    })
    
    # Kandidaten-Statistiken
    candidate_stats = {
        "count_candidates": len(df_cand),
        "candidate_churn_rate": candidate_churn_rate,
        "candidates_already_churned": int(df_cand[churn_col_numeric].sum()),
        "candidate_churn_ratio": candidate_churn_rate / overall_churn_rate if overall_churn_rate > 0 else np.nan,
    }

    return {
        "overall_statistics": {
            "overall_churn_rate": overall_churn_rate,
            "overall_churn_count": overall_churn_count,
            "overall_total": overall_total,
            "churned_sample_rate": len(df_churned) / len(df_all),
            "candidate_sample_rate": len(df_cand) / len(df_all),
            **candidate_stats,
        },
        "global_risk_features": global_top_risks,
        "candidate_overview": candidate_overview,
        "numeric_feature_profile": numeric_profile,
        "numeric_top_drivers": numeric_top_drivers,
        "categorical_feature_profile": categorical_profile,
    }

# Diese Funktion separat aufrufen oder in compute_churn_propensity_scores() einbauen
def debug_featurewise_comparison(model, df_with_keys, weight_vectors, all_vectors, churn_mask):
    """
    DEBUG 8: Feature-wise cosine gegen Centroid cosine vergleichen
    """
    DEBUG = True
    if not DEBUG:
        return
    
    # Wähle einen Test-Kunden
    for i_test in range(min(5, len(all_vectors))):
        if not all_vectors[i_test]:
            continue
            
        weight_name = list(weight_vectors.keys())[0]
        w = np.asarray(list(weight_vectors[weight_name]))
        
        # Churn-Feature-Centroids berechnen (pro Feature)
        churn_feature_centroids = []
        n_feats = len(all_vectors[0])
        
        for fi in range(n_feats):
            feat_vectors = []
            for cust_idx in np.where(churn_mask)[0]:
                if len(all_vectors[cust_idx]) > fi:
                    feat_vectors.append(all_vectors[cust_idx][fi])
            
            if feat_vectors:
                churn_feature_centroids.append(np.mean(feat_vectors, axis=0))
        
        # (B) Vorlesungsansatz: cosine pro feature, dann gewichtet mitteln
        cos_per_feat = []
        for fi in range(min(n_feats, len(all_vectors[i_test]))):
            a = all_vectors[i_test][fi]
            b = churn_feature_centroids[fi]
            cos_val = _cosine_sim(a, b)
            cos_per_feat.append(cos_val)
        
        cos_per_feat = np.asarray(cos_per_feat)
        score_weighted = float((w * cos_per_feat).sum() / (w.sum() + 1e-12))
        
        print(f"[DEBUG FEATUREWISE] i_test: {i_test}")
        print(f"[DEBUG FEATUREWISE] score_weighted (lecture): {score_weighted}")
        if cos_per_feat.size > 0:
            print(f"[DEBUG FEATUREWISE] cos_per_feat stats min/mean/max:", 
                  float(cos_per_feat.min()), float(cos_per_feat.mean()), float(cos_per_feat.max()))
            print(f"[DEBUG FEATUREWISE] top 10 contributing features (by w*cos):", 
                  np.argsort(-(w[:len(cos_per_feat)] * cos_per_feat))[:10].tolist())
        
        # Optional: Vergleich mit Ihrem aktuellen Ansatz
        from .cosine_funktions import weighted_centroid
        cust_centroid = weighted_centroid(all_vectors[i_test], w)
        
        # Churn-Centroid (gewichtet)
        churn_centroids_weighted = []
        for cust_idx in np.where(churn_mask)[0]:
            if all_vectors[cust_idx]:
                churn_centroids_weighted.append(weighted_centroid(all_vectors[cust_idx], w))
        
        if churn_centroids_weighted:
            churn_centroid_w = np.mean(churn_centroids_weighted, axis=0)
            sim_centroid = _cosine_sim(cust_centroid, churn_centroid_w)
            print(f"[DEBUG FEATUREWISE] sim_centroid (your method): {sim_centroid}")
            print(f"[DEBUG FEATUREWISE] difference: {score_weighted - sim_centroid}")
        
        break  

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
    Feature-wise Cosine Similarity Ansatz (wie in der Vorlesung).
    
    Score(x) = Σ(w_i × cos(x_i, churn_centroid_i)) / Σ(w_i)
    """
    from .cosine_funktions import weighted_centroid
    
    df_proc = df_with_keys.copy()
    
    # Churn-Spalte zu 0/1 konvertieren
    if df_proc[churn_col].dtype == object or str(df_proc[churn_col].dtype).startswith("category"):
        df_proc[churn_col] = (df_proc[churn_col] == "Churn_Yes").astype(int)
    else:
        df_proc[churn_col] = df_proc[churn_col].astype(int)
    
    churn_mask = (df_proc[churn_col].to_numpy() == 1)
    active_mask = ~churn_mask
    
    if churn_mask.sum() == 0:
        raise ValueError("Keine churned Kunden gefunden (churn=1).")
    
    # Alle Kundenvektoren laden
    all_vectors = _compute_all_customer_vectors(model, df_proc)
    n_features = len(all_vectors[0]) if all_vectors[0] else 0
    
    if n_features == 0:
        raise ValueError("Keine Feature-Vektoren gefunden.")
    
    base_df = pd.DataFrame({
        "Kunde_ID": np.arange(len(df_proc), dtype=int),
        churn_col: df_proc[churn_col].to_numpy()
    })
    
    for weight_name, weight_vec in weight_vectors.items():
        w = np.asarray(list(weight_vec), dtype=float)
        
        if len(w) != n_features:
            raise ValueError(
                f"Gewichtung '{weight_name}' hat falsche Dimension! "
                f"{len(w)} vs {n_features}"
            )
        
        # ========================================
        # FEATURE-WISE ANSATZ (WIE IN VORLESUNG)
        # ========================================
        
        # 1. Pro Feature: Centroid über alle churned Kunden
        churn_feature_centroids = []
        for feature_idx in range(n_features):
            feature_vectors = []
            for cust_idx in np.where(churn_mask)[0]:
                if len(all_vectors[cust_idx]) > feature_idx:
                    feature_vectors.append(all_vectors[cust_idx][feature_idx])
            
            if feature_vectors:
                churn_feature_centroids.append(np.mean(feature_vectors, axis=0))
            else:
                # Fallback bei fehlenden Daten
                churn_feature_centroids.append(np.zeros_like(all_vectors[0][0]))
        
        # 2. Für jeden Kunden: Feature-wise Cosine berechnen
        scores = []
        
        for cust_idx in range(len(df_proc)):
            if not all_vectors[cust_idx] or len(all_vectors[cust_idx]) != n_features:
                scores.append(np.nan)
                continue
            
            # Pro Feature: Cosine Similarity
            feature_cosines = []
            for feature_idx in range(n_features):
                customer_feature = all_vectors[cust_idx][feature_idx]
                churn_feature = churn_feature_centroids[feature_idx]
                
                cos_sim = _cosine_sim(customer_feature, churn_feature)
                feature_cosines.append(cos_sim)
            
            feature_cosines = np.array(feature_cosines)
            
            # Gewichteter Durchschnitt der Feature-Cosines
            # score = Σ(w_i × cos_i) / Σ(w_i)
            weighted_sum = np.nansum(w * feature_cosines)
            weight_sum = np.sum(w)
            
            score = weighted_sum / weight_sum if weight_sum > 0 else np.nan
            scores.append(score)
        
        scores = np.array(scores)
        
        # Churned Kunden optional maskieren
        if not include_churned:
            scores = np.where(active_mask, scores, np.nan)
        
        # Speichern
        score_col = f"ChurnProp_{weight_name}"
        base_df[score_col] = scores
        
        rank_col = f"Rank_{weight_name}"
        base_df[rank_col] = base_df[score_col].rank(ascending=False, method="min")
        
        # Z-Scores
        if zscore:
            z_col = f"ChurnPropZ_{weight_name}"
            valid = base_df[score_col].dropna()
            mu = valid.mean() if len(valid) > 0 else np.nan
            sd = valid.std(ddof=0) if len(valid) > 0 else np.nan
            
            if sd and not np.isnan(sd) and sd != 0:
                base_df[z_col] = (base_df[score_col] - mu) / sd
            else:
                base_df[z_col] = np.nan
        
        if verbose:
            v = base_df[score_col].dropna()
            if len(v) > 0:
                print(f"[FEATUREWISE] {weight_name}: "
                      f"min={float(v.min()):.4f} "
                      f"mean={float(v.mean()):.4f} "
                      f"median={float(v.median()):.4f} "
                      f"max={float(v.max()):.4f}")
    
    return base_df