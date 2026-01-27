from .cosine_funktions import (
    proximity_max,
    proximity_avg,
    proximity_topn_avg,
    subset_proximity_avg,
    proximity_avg_all
)



import os
from .key_vector_tests import get_key_values, compare_lists
from datetime import timedelta
import time
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ----- subset_proximity_avg proximity_avg_all proximity_topn_avg proximity_avg ---
def find_n_most_similar_to_key(model, key, method, n=5, df_with_keys=None):
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
        
        most_similar.append([i,avg_value])

    most_similar.sort(key=lambda x: x[1],reverse=True)
    return most_similar[:n]
        
def get_vector_list(model, key, df_with_keys=None):
    vector_list = []
    feature_names = []  # NEU: Feature-Namen speichern
    
    # DEBUG 5: Feature-Namen extrahieren (falls möglich)
    DEBUG = True
    def dprint(*args, **kwargs):
        if DEBUG:
            print(*args, **kwargs)
    
    key_values = get_key_values(key, False, prefix="key", df_with_keys=df_with_keys)

    key_values_filtered = filter_keys(key_values)
    
    # Annahme: key_values_filtered enthält Feature-Namen
    if key_values_filtered and DEBUG and key == 0:  # Nur für ersten Kunden debuggen
        dprint("[DEBUG] feature_names (key_values_filtered) length:", len(key_values_filtered))
        dprint("[DEBUG] first 20 feature_names:", key_values_filtered[:20])
    
    for value in key_values_filtered:
        vector_list.append(model.wv[value])
        feature_names.append(value)  # Speichern
    
    # Optional: Feature-Namen für späteren Zugriff speichern
    model.feature_names = feature_names

    
    return vector_list
    
def _compute_all_customer_vectors(model, df_with_keys):
    """
    Caching: Erzeugt für jeden Kunden i einmal die Vektorliste.
    Das reduziert die Rechenzeit massiv gegenüber wiederholtem get_vector_list-Aufruf.
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
    """
    Phase 1: Für jeden Referenzkunden (selected_customers.index)
    Ähnlichkeit zu allen anderen Kunden berechnen.

    Zusätzliche Parameter-Varianten:
      - S_topN_<N> für N in topn_values
      - S_k_<k>    für k in k_values

    Output (langes Format):
    Referenzkunde_ID, Kunde_ID, S_max, S_centroid, S_avg, S_topN_*, S_k_*
    """
    if df_with_keys is None or len(df_with_keys) == 0:
        raise ValueError("df_with_keys ist leer oder None.")

    if selected_customers is None or len(selected_customers) == 0:
        raise ValueError("selected_customers ist leer oder None.")

    reference_ids = list(selected_customers.index)

    # 1) Precompute: Vektorliste für alle Kunden genau einmal
    all_vectors = _compute_all_customer_vectors(model, df_with_keys)

    rows = []
    total_refs = len(reference_ids)
    total_customers = len(df_with_keys)

    # Optional: deterministische Reihenfolge + Validierung
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
                "S_centroid": proximity_avg(v_ref, v_other),
                "S_avg": proximity_avg_all(v_ref, v_other),
            }

            # S_topN für mehrere N
            for n in topn_values:
                row[f"S_topN_{n}"] = proximity_topn_avg(v_ref, v_other, n=n)

            # S_k für mehrere k (subset_size)
            for k in k_values:
                try:
                    row[f"S_k_{k}"] = subset_proximity_avg(v_ref, v_other, subset_size=k)
                except ValueError:
                    row[f"S_k_{k}"] = np.nan

            rows.append(row)

    # Spaltenreihenfolge sauber setzen
    base_cols = ["Referenzkunde_ID", "Kunde_ID", "S_max", "S_centroid", "S_avg"]
    topn_cols = [f"S_topN_{n}" for n in topn_values]
    k_cols = [f"S_k_{k}" for k in k_values]
    cols = base_cols + topn_cols + k_cols

    return pd.DataFrame(rows)[cols]



def evaluate_most_similar_proximity_avg(model, test_cases, verbose, df_with_keys=None):
    most_similar = []
    result = "\n"
    for test_case in test_cases:
        result_text = f"Test für Key_{test_case}: \t"
        result += result_text
        if verbose:
            print(result_text)
        most_similar_sentences = find_n_most_similar_to_key(model, test_case, "proximity_avg", df_with_keys=df_with_keys)
        sentence_of_testcase = get_key_values(test_case, True, prefix="key", df_with_keys=df_with_keys)
        temp = 0
        for sentence in most_similar_sentences:
            output, anzahl_paare = compare_lists(get_key_values(sentence[0], True, prefix="key", df_with_keys=df_with_keys), sentence_of_testcase)
            temp += anzahl_paare
            if verbose:
                print(f"Key_{sentence[0]}: {output}")
        result += f"{temp} gleiche Werte von 55.\n"
    return result

def evaluate_most_similar_proximity_topn_avg(model, test_cases, verbose, df_with_keys=None, method="proximity_topn_avg"):
    most_similar = []
    result = "\n"
    for test_case in test_cases:
        result_text = f"Test für Key_{test_case}: \t"
        result += result_text
        if verbose:
            print(result_text)
        most_similar_sentences = find_n_most_similar_to_key(model, test_case, method, df_with_keys=df_with_keys)
        sentence_of_testcase = get_key_values(test_case, True, prefix="key", df_with_keys=df_with_keys)
        temp = 0
        for sentence in most_similar_sentences:
            output, anzahl_paare = compare_lists(get_key_values(sentence[0], True, prefix="key", df_with_keys=df_with_keys), sentence_of_testcase)
            temp += anzahl_paare
            if verbose:
                print(f"Key_{sentence[0]}: {output}")
        result += f"{temp} gleiche Werte von 55.\n"
    return result

def find_global_most_similar(model, verbose, n=5, df_with_keys=None):
    """
    Vereinfachte Version mit grundlegender Fortschrittsanzeige
    """
    all_similarities = []
    total_cases = len(df_with_keys)
    total_iterations = total_cases * (total_cases - 1) // 2
    iterations_done = 0
    
    start_time = time.time()
    
    # Pre-load vectors
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
        


        # PHASE 1
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
            # Neu berechnen
            phase1_df = compute_phase1_similarity_df(
                model=model,
                df_with_keys=df_with_keys,
                selected_customers=selected_customers,
                topn_values=(2, 5, 10),
                k_values=(3, 5, 7),
                verbose=verbose
            )
            
            # Speichern
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
        
        # PHASE 2: Rankings
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
        
        # PHASE 2: Overlaps
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

                # PHASE 3: Statistische Charakterisierung
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


        # PHASE 4: Diskriminierungsfähigkeit (Separation Score)
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
    """
    Erstellt Top-200 und Bottom-200 Rankings für jede Metrik pro Referenzkunde.
    
    Parameters:
    -----------
    phase1_df : pd.DataFrame
        DataFrame aus Phase 1 mit Spalten: Referenzkunde_ID, Kunde_ID, S_max, S_centroid, ...
    top_n : int
        Anzahl der Top/Bottom Elemente (default: 200)
    verbose : bool
        Ausgabe von Fortschrittsinformationen
    
    Returns:
    --------
    dict : Dictionary mit Rankings in strukturierter Form
        Struktur: {
            'Referenzkunde_ID': {
                'top': {
                    'S_max': [Kunde_ID_1, Kunde_ID_2, ...],  # Top-200
                    'S_centroid': [...],
                    ...
                },
                'bottom': {
                    'S_max': [Kunde_ID_1, Kunde_ID_2, ...],  # Bottom-200  
                    'S_centroid': [...],
                    ...
                }
            },
            ...
        }
    """
    # Alle Ähnlichkeits-Metriken identifizieren (außer ID-Spalten)
    metric_cols = [col for col in phase1_df.columns 
                   if col not in ['Referenzkunde_ID', 'Kunde_ID']]
    
    # Alle Referenzkunden-IDs
    reference_ids = phase1_df['Referenzkunde_ID'].unique()
    
    rankings_dict = {}
    
    for ref_id in reference_ids:
        if verbose:
            print(f"Erstelle Rankings für Referenzkunde {ref_id}")
        
        # Daten für diesen Referenzkunden filtern
        ref_data = phase1_df[phase1_df['Referenzkunde_ID'] == ref_id].copy()
        
        ref_dict = {'top': {}, 'bottom': {}}
        
        for metric in metric_cols:
            # Aufsteigend sortieren für Bottom-200 (niedrigste Ähnlichkeit = unterschiedlichste)
            bottom_sorted = ref_data.sort_values(by=metric, ascending=True)
            bottom_ranking = bottom_sorted['Kunde_ID'].head(top_n).tolist()
            
            # Absteigend sortieren für Top-200 (höchste Ähnlichkeit = ähnlichste)
            top_sorted = ref_data.sort_values(by=metric, ascending=False)
            top_ranking = top_sorted['Kunde_ID'].head(top_n).tolist()
            
            ref_dict['top'][metric] = top_ranking
            ref_dict['bottom'][metric] = bottom_ranking
        
        rankings_dict[ref_id] = ref_dict
    
    return rankings_dict




def compute_overlap_coefficients(rankings_dict, verbose=False):
    """
    Berechnet Overlap-Koeffizienten zwischen allen Metrik-Paaren.
    
    Parameters:
    -----------
    rankings_dict : dict
        Dictionary aus create_rankings_for_phase2()
    verbose : bool
        Ausgabe von Fortschrittsinformationen
    
    Returns:
    --------
    dict : Dictionary mit Overlap-Koeffizienten in strukturierter Form
        Struktur: {
            'Referenzkunde_ID': {
                'top_overlaps': pd.DataFrame,  # Overlaps zwischen Top-Rankings
                'bottom_overlaps': pd.DataFrame, # Overlaps zwischen Bottom-Rankings
                'summary_stats': pd.DataFrame   # Zusammenfassende Statistiken
            },
            ...
        }
    """
    from itertools import combinations
    
    results_dict = {}
    
    for ref_id, rankings in rankings_dict.items():
        if verbose:
            print(f"Berechne Overlaps für Referenzkunde {ref_id}")
        
        # Metriken extrahieren
        metrics = list(rankings['top'].keys())
        
        # DataFrames für Overlaps vorbereiten
        top_overlap_df = pd.DataFrame(index=metrics, columns=metrics)
        bottom_overlap_df = pd.DataFrame(index=metrics, columns=metrics)
        
        # Alle Metrik-Paare durchgehen
        for metric1, metric2 in combinations(metrics, 2):
            # Top-200 Overlap berechnen
            top_set1 = set(rankings['top'][metric1])
            top_set2 = set(rankings['top'][metric2])
            top_overlap = len(top_set1.intersection(top_set2)) / min(len(top_set1), len(top_set2))
            
            # Bottom-200 Overlap berechnen
            bottom_set1 = set(rankings['bottom'][metric1])
            bottom_set2 = set(rankings['bottom'][metric2])
            bottom_overlap = len(bottom_set1.intersection(bottom_set2)) / min(len(bottom_set1), len(bottom_set2))
            
            # In DataFrames eintragen (symmetrisch)
            top_overlap_df.loc[metric1, metric2] = top_overlap
            top_overlap_df.loc[metric2, metric1] = top_overlap
            bottom_overlap_df.loc[metric1, metric2] = bottom_overlap
            bottom_overlap_df.loc[metric2, metric1] = bottom_overlap
        
        # Diagonale auf 1.0 setzen (Vergleich einer Metrik mit sich selbst)
        for metric in metrics:
            top_overlap_df.loc[metric, metric] = 1.0
            bottom_overlap_df.loc[metric, metric] = 1.0
        
        # Statistiken berechnen
        summary_rows = []
        for metric in metrics:
            # Durchschnittlicher Overlap dieser Metrik mit allen anderen
            top_avg = top_overlap_df.loc[metric, top_overlap_df.columns != metric].mean()
            bottom_avg = bottom_overlap_df.loc[metric, bottom_overlap_df.columns != metric].mean()
            
            summary_rows.append({
                'Metrik': metric,
                'Top_Avg_Overlap': top_avg,
                'Bottom_Avg_Overlap': bottom_avg,
                'Overlap_Diff': top_avg - bottom_avg  # Interessant für Analyse
            })
        
        summary_df = pd.DataFrame(summary_rows)
        
        # Ergebnisse speichern
        results_dict[ref_id] = {
            'top_overlaps': top_overlap_df,
            'bottom_overlaps': bottom_overlap_df,
            'summary_stats': summary_df
        }
        
        if verbose:
            print(f"  Top-Overlap-Matrix:\n{top_overlap_df.round(3)}\n")
            print(f"  Zusammenfassung:\n{summary_df.round(3)}\n")
    
    return results_dict





def get_diverse_customers_with_clusters(df, k=5, random_state=42, verbose = False):
    """
    Findet 5 diverse Kunden UND zeigt, welche Kundengruppe sie vertreten
    """
    # 1. Daten vorbereiten (kategorische Features encoden)
    df_encoded = prepare_data_for_clustering(df, verbose)
    
    # 2. Features skalieren für besseres Clustering
    features = [col for col in df_encoded.columns if col not in ['key_1', 'key_2', 'key3', 'key4', 'key5', 'key6']]
    X = df_encoded[features].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. K-Means Clustering
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # 4. Cluster-Zentren analysieren (für Cluster-Beschreibung)
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    
    # 5. Aus jedem Cluster einen repräsentativen Kunden wählen
    selected_indices = []
    selected_cluster_info = []
    
    for cluster_id in range(k):
        # Indizes aller Kunden in diesem Cluster
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        
        if len(cluster_indices) > 0:
            # Wähle den Kunden, der am weitesten vom Cluster-Zentrum entfernt ist
            # (repräsentiert die Vielfalt innerhalb des Clusters)
            cluster_points = X_scaled[cluster_indices]
            cluster_center = kmeans.cluster_centers_[cluster_id]
            
            # Distanzen berechnen
            distances = np.linalg.norm(cluster_points - cluster_center, axis=1)
            
            # Den am weitesten entfernten Punkt wählen
            farthest_idx_in_cluster = np.argmax(distances)
            selected_idx = cluster_indices[farthest_idx_in_cluster]
            
            selected_indices.append(selected_idx)
            
            # Cluster-Charakteristiken speichern
            cluster_info = {
                'cluster_id': cluster_id,
                'cluster_size': len(cluster_indices),
                'cluster_center_original': cluster_centers[cluster_id],
                'selected_customer_idx': selected_idx
            }
            selected_cluster_info.append(cluster_info)
    
    # 6. Ergebnisse zusammenstellen
    selected_customers = df.iloc[selected_indices].copy()
    
    # Cluster-Labels hinzufügen
    selected_customers['cluster_group'] = [info['cluster_id'] for info in selected_cluster_info]
    selected_customers['cluster_size'] = [info['cluster_size'] for info in selected_cluster_info]
    
    return selected_customers, cluster_centers, selected_cluster_info, kmeans


def prepare_data_for_clustering(df, verbose = False):
    """
    Bereitet Daten für Clustering vor:
    - Entfernt redundante Schlüsselspalten
    - Encodet kategorische Variablen
    - Behandelt fehlende Werte
    """
    # Erstelle eine Kopie
    df_clean = df.copy()
    
    if verbose:
        print("Schritt 1: Redundante Spalten entfernen...")
    
    # Entferne alle Duplikat-Schlüsselspalten
    # Basierend auf Ihrer Struktur: key_1, key_2, key3, key4, key5, key6
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
    
    # Basierend auf Ihrem Beispiel sind dies die kategorischen Spalten
    categorical_columns = [
        'Credit_Score',  # Wird sein wie "Credit_Score_Fair"
        'Country',       # Wird sein wie "Country_France"
        'gender',        # Gender_Female/Male
        'age',           # Age_Adults_in_their_Prime
        'tenure',        # Tenure_1, Tenure_2, etc.
        'balance',       # Balance_Cluster_1, etc.
        'products_number', # ProductsNumber_1, etc.
        'credit_card',   # CreditCard_Yes/No
        'active_member', # ActiveMember_Yes/No
        'estimated_salary', # Salary_High, etc.
        'churn'          # Churn_Yes/No
    ]
    
    # Finde tatsächliche Spaltennamen im DataFrame
    actual_categorical_cols = []
    for pattern in categorical_columns:
        matching = [col for col in df_clean.columns if pattern.lower() in col.lower()]
        actual_categorical_cols.extend(matching)
    
    if verbose:
        print(f"Gefundene kategorische Spalten: {actual_categorical_cols}")
    
    if verbose:
        print("\nSchritt 3: One-Hot Encoding durchführen...")
    
    # Für jede kategorische Spalte: One-Hot Encoding
    encoded_dfs = []
    
    for col in actual_categorical_cols:
        # Prüfe, ob die Spalte bereits binär ist
        unique_vals = df_clean[col].astype(str).unique()
        
        if len(unique_vals) == 2:
            # Binäre Spalte: Konvertiere zu 0/1
            mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
            encoded = df_clean[col].map(mapping)
            encoded_dfs.append(pd.DataFrame({col: encoded}))
        else:
            # Reguläres One-Hot Encoding
            # Extrahiere Basisnamen (z.B. "Credit_Score" aus "Credit_Score_Fair")
            base_name = col.split('_')[0] if '_' in col else col
            
            # Für Spalten, die Werte wie "Category_Value" enthalten
            if df_clean[col].astype(str).str.contains('_').any():
                # Spalte enthält selbst encodierte Werte
                # Erstelle Dummy-Variablen
                dummies = pd.get_dummies(df_clean[col], prefix=base_name)
                encoded_dfs.append(dummies)
            else:
                # Normale kategorische Spalte
                dummies = pd.get_dummies(df_clean[col], prefix=col)
                encoded_dfs.append(dummies)
    
    # Numerische Spalten finden (falls vorhanden)
    numeric_cols = []
    for col in df_clean.columns:
        if col not in actual_categorical_cols:
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                numeric_cols.append(col)
    
    if verbose:
        print(f"Numerische Spalten: {numeric_cols}")
    
    # Füge numerische Spalten hinzu
    if numeric_cols:
        encoded_dfs.append(df_clean[numeric_cols])
    
    # Alles zusammenführen
    result = pd.concat(encoded_dfs, axis=1)
    
    if verbose:
        print(f"\nErgebnis: {result.shape[1]} Features nach Encoding")
    
    return result

def interpret_clusters(selected_customers, cluster_centers, selected_cluster_info, 
                       df_original, kmeans, feature_names=None, verbose=True):
    """
    Interpretiert die Cluster-Informationen aus get_diverse_customers_with_clusters
    
    Parameters:
    -----------
    selected_customers : DataFrame
        Ausgabe von get_diverse_customers_with_clusters
    cluster_centers : ndarray
        Cluster-Zentren im originalen Feature-Raum
    selected_cluster_info : list
        Liste mit Cluster-Informationen
    df_original : DataFrame
        Originaler DataFrame (vor Encoding)
    kmeans : KMeans model
        Trainiertes KMeans-Modell
    feature_names : list, optional
        Namen der Features nach Encoding
    verbose : bool
        Ob ausführliche Ausgabe gewünscht ist
    """
    
    if verbose:
        print("=" * 70)
        print("CLUSTER-INTERPRETATION")
        print("=" * 70)
    
    # 1. Grundlegende Cluster-Informationen
    cluster_summary = []
    
    for info in selected_cluster_info:
        cluster_id = info['cluster_id']
        cluster_size = info['cluster_size']
        center = info['cluster_center_original']
        
        # Den ausgewählten Kunden für diesen Cluster finden
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
    
    # 2. Feature-Namen extrahieren (falls nicht gegeben)
    if feature_names is None:
        # Versuche, Feature-Namen zu rekonstruieren
        # Zuerst von den numerischen Spalten nach Encoding
        encoded_cols = []
        for col in selected_customers.columns:
            if 'cluster' not in col.lower() and 'key' not in col.lower():
                # Dies ist ein Original-Feature
                encoded_cols.append(col)
        
        # Für One-Hot encoded Features müssen wir gruppieren
        feature_groups = {}
        for col in encoded_cols:
            # Versuche Basisnamen zu extrahieren (z.B. "Credit_Score" aus "Credit_Score_Fair")
            if '_' in col:
                base = col.split('_')[0]
                if base not in feature_groups:
                    feature_groups[base] = []
                feature_groups[base].append(col)
        
        feature_names = []
        for base, cols in feature_groups.items():
            if len(cols) > 1:
                # One-Hot encoded Feature Gruppe
                feature_names.extend(cols)
            else:
                # Einzelnes Feature
                feature_names.append(cols[0])
    
    if verbose:
        print(f"\nGefundene {len(cluster_summary)} Cluster:")
        print("-" * 70)
    
    # 3. Detaillierte Cluster-Analyse
    detailed_interpretations = []
    
    for summary in cluster_summary:
        cluster_id = summary['cluster_id']
        
        if verbose:
            print(f"\n📊 CLUSTER {cluster_id}")
            print(f"   Größe: {summary['size']} Kunden ({summary['percentage']:.1f}%)")
            print(f"   Repräsentant: {summary['customer_key']}")
        
        # Cluster-Zentrum analysieren
        center = summary['center_values']
        
        # Finde die 5 wichtigsten/auffälligsten Features für diesen Cluster
        if len(center) > 0:
            # Annahme: Große absolute Werte im Zentrum sind charakteristisch
            abs_center = np.abs(center)
            
            # Für One-Hot Features müssen wir anders vorgehen
            # Finde Features mit hohen positiven Werten (näher an 1)
            high_value_indices = np.where(center > 0.5)[0]
            
            if len(high_value_indices) > 0:
                # Sortiere nach Wert (absteigend)
                sorted_indices = high_value_indices[np.argsort(-center[high_value_indices])]
                
                # Beschränke auf Top 5
                top_indices = sorted_indices[:5]
                
                interpretation = {
                    'cluster_id': cluster_id,
                    'top_features': [],
                    'characteristics': []
                }
                
                if verbose:
                    print(f"   Charakteristische Merkmale:")
                
            
        
        # Kunden in diesem Cluster analysieren
        cluster_customers = df_original.iloc[
            np.where(kmeans.labels_ == cluster_id)[0]
        ]
        
        if verbose and len(cluster_customers) > 0:
            # Typische Werte für einige wichtige Spalten
            important_cols = ['Credit_Score', 'Country', 'age', 'estimated_salary', 'balance']
            available_cols = [col for col in important_cols if col in cluster_customers.columns]
            
            print(f"   Typische Werte in dieser Gruppe:")
            for col in available_cols[:3]:  # Nur erste 3 zeigen
                if col in cluster_customers.columns:
                    # Häufigster Wert
                    most_common = cluster_customers[col].mode()
                    if not most_common.empty:
                        value = most_common.iloc[0]
                        # Kürze lange Strings
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
    """
    Zentrale Funktion zum Speichern oder Laden von Phasenergebnissen.
    Verwendet NumPy .npz Format wie in deinem Beispiel.
    
    Parameters:
    -----------
    phase_name : str
        Name der Phase ('phase1_similarity', 'phase2_rankings', 'phase2_overlaps')
    phase_data : object
        Die zu speichernden Daten (DataFrame, dict, etc.)
    base_dir : str
        Basisverzeichnis
    model_idx : int
        Index des aktuellen Models in der Liste
    model_info : dict
        Modellinformationen
    selected_customers : pd.DataFrame
        DataFrame mit ausgewählten Kunden (für Dateinamen)
    verbose : bool
        Ausgabe von Statusmeldungen
    
    Returns:
    --------
    object : Geladene oder neu berechnete Daten
    """
    # Extrahiere die IDs der ausgewählten Kunden für den Dateinamen
    customer_ids = "_".join(str(idx) for idx in selected_customers.index.tolist())
    
    # Baue den Dateinamen auf (genau wie in deinem Beispiel)
    filename = f"{base_dir}/precalculated_benchmark_values/{phase_name}_"
    
    # Füge Modellparameter hinzu
    for key, value in model_info.items():
        if isinstance(value, (str, int, float)):
            filename += f"_{key[:3]}_{value}"
    
    # Füge Model-Index und Kunden-IDs hinzu
    filename += f"_model{model_idx}_customers_{customer_ids}.npz"
    
    # Prüfe, ob Datei existiert
    if os.path.exists(filename):
        if verbose:
            print(f"Lade {phase_name} von {filename}")
        
        # Datei laden
        loaded_data = np.load(filename, allow_pickle=True)
        
        
        # Je nach Phase die Daten unterschiedlich rekonstruieren
        if phase_name == "phase1_similarity":
            # DataFrame rekonstruieren
            df = pd.DataFrame({
                'Referenzkunde_ID': loaded_data['referenz_ids'],
                'Kunde_ID': loaded_data['kunde_ids']
            })
            
            # Metriken-Spalten hinzufügen
            metric_names = loaded_data['metric_names']
            for i, metric_name in enumerate(metric_names):
                df[metric_name] = loaded_data[f'metric_{i}']
            
            return df
        elif phase_name in ("phase3_stats", "phase4_separation"):
            loaded_obj = loaded_data["payload"].item()
            return loaded_obj

        
        elif phase_name == "phase2_rankings":
            # Dictionary rekonstruieren
            rankings_dict = {}
            ref_ids = loaded_data['ref_ids']
            
            for ref_id in ref_ids:
                ref_id_str = str(ref_id)
                rankings_dict[ref_id] = {
                    'top': {},
                    'bottom': {}
                }
                
                # Metriken für diesen Referenzkunden
                metrics = loaded_data[f'{ref_id_str}_metrics']
                
                for metric in metrics:
                    # Top-Rankings
                    top_key = f'{ref_id_str}_top_{metric}'
                    if top_key in loaded_data:
                        rankings_dict[ref_id]['top'][metric] = loaded_data[top_key].tolist()
                    
                    # Bottom-Rankings
                    bottom_key = f'{ref_id_str}_bottom_{metric}'
                    if bottom_key in loaded_data:
                        rankings_dict[ref_id]['bottom'][metric] = loaded_data[bottom_key].tolist()
            
            return rankings_dict
        
        elif phase_name == "phase2_overlaps":
            # Dictionary rekonstruieren
            overlaps_dict = {}
            ref_ids = loaded_data['ref_ids']
            
            for ref_id in ref_ids:
                ref_id_str = str(ref_id)
                ref_key = f'{ref_id_str}_data'
                
                if ref_key in loaded_data:
                    ref_data = loaded_data[ref_key].item()  # .item() für pickle-Objekte in npz
                    overlaps_dict[ref_id] = ref_data
            
            return overlaps_dict
        elif phase_name in ("phase3_stats", "phase4_separation"):
            # generisches Speichern als Pickle payload
            save_dict = {
                "payload": np.array([phase_data], dtype=object)
            }
            np.savez(filename, **save_dict, allow_pickle=True)

    
    # Falls nicht existiert und phase_data=None: zurückgeben, damit neu berechnet wird
    if phase_data is None:
        return None
    
    # Falls nicht existiert: speichere die Daten
    if verbose:
        print(f"Speichere {phase_name} in {filename}")
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Je nach Datentyp speichern
    if phase_name == "phase1_similarity" and isinstance(phase_data, pd.DataFrame):
        # DataFrame speichern
        save_dict = {
            'referenz_ids': phase_data['Referenzkunde_ID'].to_numpy(),
            'kunde_ids': phase_data['Kunde_ID'].to_numpy()
        }
        
        # Metriken extrahieren (alle Spalten außer ID-Spalten)
        metric_cols = [col for col in phase_data.columns 
                      if col not in ['Referenzkunde_ID', 'Kunde_ID']]
        save_dict['metric_names'] = np.array(metric_cols, dtype=object)
        
        # Jede Metrik-Spalte speichern
        for i, metric in enumerate(metric_cols):
            save_dict[f'metric_{i}'] = phase_data[metric].to_numpy()
        
        np.savez(filename, **save_dict)
        
    elif phase_name == "phase2_rankings" and isinstance(phase_data, dict):
        # Rankings-Dictionary speichern
        save_dict = {
            'ref_ids': np.array(list(phase_data.keys()))
        }
        
        for ref_id, ref_data in phase_data.items():
            ref_id_str = str(ref_id)
            metrics = list(ref_data['top'].keys())
            save_dict[f'{ref_id_str}_metrics'] = np.array(metrics, dtype=object)
            
            for metric in metrics:
                # Top-Rankings
                save_dict[f'{ref_id_str}_top_{metric}'] = np.array(ref_data['top'][metric])
                # Bottom-Rankings
                save_dict[f'{ref_id_str}_bottom_{metric}'] = np.array(ref_data['bottom'][metric])
        
        np.savez(filename, **save_dict)
        
    elif phase_name == "phase2_overlaps" and isinstance(phase_data, dict):
        # Overlaps-Dictionary speichern
        save_dict = {
            'ref_ids': np.array(list(phase_data.keys()))
        }
        
        for ref_id, ref_data in phase_data.items():
            ref_id_str = str(ref_id)
            # Speichere das gesamte Dictionary für diesen Referenzkunden
            # (DataFrames werden als pickle in npz gespeichert)
            save_dict[f'{ref_id_str}_data'] = np.array([ref_data], dtype=object)
        
        np.savez(filename, **save_dict, allow_pickle=True)
    
    return phase_data


def compute_phase3_metric_statistics(
    phase1_df: pd.DataFrame,
    bins: int = 50,
    use_global_bins_across_metrics: bool = True
) -> dict:
    """
    Phase 3: Statistische Charakterisierung der Metriken.

    Berechnet je Metrik:
      - mean, median, std, min, max, range
      - count_negative, count_total, share_negative
      - Histogramm (counts + bin_edges) mit gleicher Bin-Anzahl.
    
    Parameter:
      bins: gleiche Anzahl Bins für alle Metriken.
      use_global_bins_across_metrics:
        - True: gemeinsame bin_edges über ALLE Metriken (direkt vergleichbare Histogramme)
        - False: bin_edges je Metrik separat (trotz gleicher Bin-Anzahl)

    Returns:
      {
        "stats_df": pd.DataFrame,
        "histograms": {metric: {"bin_edges": np.ndarray, "counts": np.ndarray}}
        "bins": int,
        "use_global_bins_across_metrics": bool
      }
    """
    if phase1_df is None or len(phase1_df) == 0:
        raise ValueError("phase1_df ist leer oder None.")

    metric_cols = [c for c in phase1_df.columns if c not in ["Referenzkunde_ID", "Kunde_ID"]]
    if not metric_cols:
        raise ValueError("Keine Metrikspalten gefunden (erwartet: alles außer Referenzkunde_ID, Kunde_ID).")

    # Alle Werte (flatten) zur Bestimmung globaler Bins (optional)
    all_values = []
    for m in metric_cols:
        s = pd.to_numeric(phase1_df[m], errors="coerce").dropna()
        if len(s) > 0:
            all_values.append(s.to_numpy())
    if not all_values:
        raise ValueError("Alle Metrikspalten enthalten nur NaN/nicht-numerische Werte.")

    all_values = np.concatenate(all_values)

    # Bin-Kanten definieren
    if use_global_bins_across_metrics:
        global_min = np.nanmin(all_values)
        global_max = np.nanmax(all_values)
        if np.isclose(global_min, global_max):
            # degenerierter Fall: alle Werte gleich
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

        # Histogramm
        if use_global_bins_across_metrics:
            counts, bin_edges = np.histogram(arr, bins=bin_edges_global)
        else:
            # m-spezifische edges, aber gleiche Bin-Anzahl
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
    """
    Phase 4: Diskriminierungsfähigkeit (Separation Score).

    Für jede Metrik und jeden Referenzkunden:
      Separation = mean(similarity(top)) - mean(similarity(bottom))

    rankings_dict ist die Ausgabe von create_rankings_for_phase2().
    top_n wird hier nur zur Plausibilisierung genutzt (Rankings sollten bereits top_n enthalten).

    Returns:
      {
        "per_reference_df": pd.DataFrame,  # Referenzkunde_ID, Metrik, mean_top, mean_bottom, separation
        "summary_df": pd.DataFrame         # Metrik, separation_mean, separation_median, separation_std
      }
    """
    if phase1_df is None or len(phase1_df) == 0:
        raise ValueError("phase1_df ist leer oder None.")
    if rankings_dict is None or len(rankings_dict) == 0:
        raise ValueError("rankings_dict ist leer oder None.")

    metric_cols = [c for c in phase1_df.columns if c not in ["Referenzkunde_ID", "Kunde_ID"]]
    if not metric_cols:
        raise ValueError("Keine Metrikspalten gefunden (erwartet: alles außer Referenzkunde_ID, Kunde_ID).")

    # Schneller Lookup: je Referenzkunde eine Untertabelle indexiert nach Kunde_ID
    # -> vermeidet wiederholte Filter-/Merge-Kosten
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

        # Metriken für diesen Referenzkunden (aus Rankings)
        metrics = list(top_dict.keys())

        for m in metrics:
            if m not in metric_cols:
                continue

            top_ids = top_dict.get(m, [])[:top_n]
            bottom_ids = bottom_dict.get(m, [])[:top_n]

            # Werte ziehen; fehlende IDs -> NaN
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
    """
    Filtert alle 'key_'-Werte heraus, behält aber das erste Vorkommen.
    
    Parameters:
    -----------
    values : list
        Liste von Strings
    
    Returns:
    --------
    list
        Gefilterte Liste
    """
    filtered = []
    first_key_found = False
    
    for value in values:
        if value.startswith('key_'):
            if not first_key_found:
                # Behalte das erste Vorkommen
                filtered.append(value)
                first_key_found = True
            # Alle weiteren key_-Werte werden übersprungen
        else:
            # Alle anderen Werte behalten
            filtered.append(value)
    
    return filtered