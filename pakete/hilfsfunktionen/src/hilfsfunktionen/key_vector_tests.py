import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set

# ------- Deinition of Tests to compare Key-Vectors -----------
def get_key_values(key, filter, prefix="key", df_with_keys=None):
    values_with_keys = df_with_keys.iloc[key] #mit Keys
    if filter:
        return [w for w in values_with_keys if not w.startswith(prefix)]
    else:
        return [w for w in values_with_keys]

def compare_lists(list1, list2):
    output:str = ""
    anzahl_paare:int = 0
    gleiche_paare = [(i, a, b) for i, (a, b) in enumerate(zip(list1, list2)) if a == b]
    anzahl_paare = len(gleiche_paare)
    output += f"{anzahl_paare} von {len(list1)} Gleiche Werte: "
    for pos, a, b in gleiche_paare:
        output += f"(Position {pos} : '{a}' )"
    return output, anzahl_paare

def has_churned(customer_index: int, df: pd.DataFrame):
    """
    Check if a customer has turned based on their row index.

    Prameters:
    ----------
    customer_index : int
        Zero_based index of constomer row in the dataframe.
    df : pd.DataFrame
        DataFrame containing cusomer data with 'churn'-column
        
    Returns:
    --------
    bool
        True if customer has churned (churn == 'Churn_Yes'), False otherwise

    Raises:
    -------
    IndexError
        if customer_index out of bounds
    ColumnError
        If 'churn' column does not exist
    """
    if customer_index < 0 or customer_index >= len(df):
        raise IndexError(f"Customer index {customer_index} out of bounds for [0, {len(df-1)}]")
    if 'churn' not in df.columns:
        raise IndexError(f"Column 'churn' not in the DataFrame.")

    churned_value = df.iloc[customer_index]['churn']

    if pd.isna(churned_value):
        return False
    
    return "Churn_Yes" == churned_value


# ------- Hilfsfunktionen -----------

def get_key_values_new(key: int, filter_keys: bool = True, prefix: str = "key", df_with_keys=None) -> List[str]:
    """Extrahiert Werte aus einer Zeile basierend auf dem Key-Index."""
    values_with_keys = df_with_keys.iloc[key]
    if filter_keys:
        return [w for w in values_with_keys if not w.startswith(prefix)]
    else:
        return [w for w in values_with_keys]

# ---------------------------------------------------
# Hauptmetrik: Key-zu-Zeilenwerte Proximity Score
# ---------------------------------------------------

def calculate_key_to_values_score(model, key: str, df_with_keys, topn: int = 20, 
                                   filter_keys: bool = True, max_fetch: int = 1000, 
                                   verbose: bool = False) -> Dict:
    """
    Misst, wie gut ein Key die semantische Information seiner Zeile kodiert.
    
    Die Hypothese: Ein guter Key sollte im Vektorraum nahe bei seinen eigenen
    Zeilenwerten liegen (z.B. key_1 nahe bei Age_Young_Seniors, Gender_Male, etc.)
    
    Parameters:
    -----------
    model : Word2Vec model
    key : str (z.B. "key_1")
    df_with_keys : pd.DataFrame
    topn : int - Anzahl der gewünschten Nicht-Key Nachbarn
    filter_keys : bool - Wenn True, werden andere Keys aus den Nachbarn gefiltert
    max_fetch : int - Maximale Anzahl initialer Nachbarn (Standard: 1000)
                      Wird nur verwendet wenn filter_keys=True
    verbose : bool - Detaillierte Ausgabe
    
    Returns:
    --------
    Dict mit:
        - 'proximity_score': Anteil der eigenen Zeilenwerte in Top-N Nachbarn (0.0-1.0)
        - 'weighted_proximity_score': Gewichteter Score (nähere Nachbarn zählen mehr)
        - 'found_values': Anzahl gefundener eigener Zeilenwerte
        - 'total_values': Gesamtzahl der Zeilenwerte
        - 'missing_values': Zeilenwerte die nicht in Top-N sind
        - 'details': Detaillierte Informationen
    """
    key_index = int(key.replace("key_", ""))
    key_values = get_key_values_new(key_index, filter_keys=True, df_with_keys=df_with_keys)
    
    # Filtere Zeilenwerte die im Modell-Vokabular sind
    key_values_in_vocab = [v for v in key_values if v in model.wv]
    
    if len(key_values_in_vocab) == 0:
        return {
            'error': f'Keine Zeilenwerte von {key} im Modell-Vokabular',
            'key': key,
            'total_values': len(key_values)
        }
    
    try:
        if filter_keys:
            # Hole deutlich mehr Nachbarn als nötig, um nach Filterung genug zu haben
            all_neighbors = model.wv.most_similar(key, topn=max_fetch)
            
            # Filtere alle Keys raus
            non_key_neighbors = [(word, sim) for word, sim in all_neighbors 
                                  if not word.startswith("key_")]
            
            # Nimm die Top-N der gefilterten Nachbarn
            neighbors = non_key_neighbors[:topn]
            
            if verbose:
                total_fetched = len(all_neighbors)
                keys_filtered = total_fetched - len(non_key_neighbors)
                actual_retrieved = len(neighbors)
                
                print(f"ℹ️  Nachbar-Statistik:")
                print(f"   Initial geholt:        {total_fetched}")
                print(f"   Davon Keys gefiltert:  {keys_filtered} ({keys_filtered/total_fetched*100:.1f}%)")
                print(f"   Verbleibende Nicht-Keys: {len(non_key_neighbors)}")
                print(f"   Verwendete Top-N:      {actual_retrieved}")
                
                if actual_retrieved < topn:
                    print(f"   ⚠️  Warnung: Nur {actual_retrieved}/{topn} Nicht-Key-Nachbarn gefunden!")
                    print(f"      Erhöhe max_fetch (aktuell {max_fetch}) für bessere Ergebnisse.")
        else:
            # Ohne Filter: Hole direkt die gewünschte Anzahl
            neighbors = model.wv.most_similar(key, topn=topn)
            
    except KeyError:
        return {'error': f'{key} nicht im Vokabular'}
    
    # Erstelle Set der Nachbar-Wörter für schnelle Suche
    neighbor_words = {word for word, _ in neighbors}
    
    # Finde welche eigenen Zeilenwerte in den Nachbarn sind
    found_values = []
    missing_values = []
    
    for value in key_values_in_vocab:
        if value in neighbor_words:
            # Finde Rang und Similarity
            rank = next(i for i, (w, _) in enumerate(neighbors) if w == value)
            similarity = neighbors[rank][1]
            found_values.append({
                'value': value,
                'rank': rank,
                'similarity': similarity,
                'weight': 1.0 / (rank + 1)
            })
        else:
            # Prüfe wie weit entfernt der Wert tatsächlich ist
            try:
                actual_similarity = model.wv.similarity(key, value)
                missing_values.append({
                    'value': value,
                    'similarity': actual_similarity
                })
            except KeyError:
                missing_values.append({
                    'value': value,
                    'similarity': None
                })
    
    # Berechne Scores
    proximity_score = len(found_values) / len(key_values_in_vocab)
    
    # Gewichteter Score: Werte die näher am Key sind, zählen mehr
    if len(found_values) > 0:
        weights = [item['weight'] for item in found_values]
        # Normalisiere Gewichte
        weight_sum = sum(weights)
        weighted_proximity_score = weight_sum / len(key_values_in_vocab)
    else:
        weighted_proximity_score = 0.0
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Analyse für {key} (Zeile {key_index})")
        print(f"{'='*60}")
        print(f"Filter Keys: {'Ja' if filter_keys else 'Nein'}")
        if filter_keys:
            print(f"Max Fetch: {max_fetch}")
        print(f"Zeilenwerte im Vokabular: {len(key_values_in_vocab)}/{len(key_values)}")
        print(f"\nGefundene Werte in Top-{len(neighbors)} Nachbarn: {len(found_values)}/{len(key_values_in_vocab)}")
        
        if found_values:
            print("\n✓ Gefundene Zeilenwerte:")
            for item in sorted(found_values, key=lambda x: x['rank']):
                print(f"  Rang {item['rank']:2d}: {item['value']:30s} (Sim: {item['similarity']:.4f})")
        
        if missing_values and verbose:
            print(f"\n✗ Fehlende Zeilenwerte (nicht in Top-{topn}):")
            for item in sorted(missing_values, key=lambda x: x['similarity'] if x['similarity'] else -1, reverse=True)[:5]:
                sim_str = f"{item['similarity']:.4f}" if item['similarity'] else "N/A"
                print(f"  {item['value']:30s} (Sim: {sim_str})")
        
        print(f"\nProximity Score:          {proximity_score:.3f}")
        print(f"Weighted Proximity Score: {weighted_proximity_score:.3f}")
    
    return {
        'key': key,
        'proximity_score': proximity_score,
        'weighted_proximity_score': weighted_proximity_score,
        'found_values': len(found_values),
        'total_values_in_vocab': len(key_values_in_vocab),
        'total_values': len(key_values),
        'missing_values': len(missing_values),
        'found_details': found_values,
        'missing_details': missing_values,
        'topn': topn,
        'filter_keys': filter_keys,
        'max_fetch': max_fetch if filter_keys else None,
        'actual_neighbors_count': len(neighbors)
    }

# ---------------------------------------------------
# Gesamtbewertung über mehrere Keys
# ---------------------------------------------------

def evaluate_model_key_quality(model, df_with_keys, num_keys: int = 10, topn: int = 20, 
                                filter_keys: bool = True, max_fetch: int = 1000, 
                                verbose: bool = False) -> Dict:
    """
    Evaluiert die Key-Quality über mehrere Keys.
    
    Misst wie gut Keys die semantische Information ihrer Zeilen kodieren,
    indem geprüft wird, ob die Zeilenwerte im Vektorraum nahe beim Key liegen.
    
    Parameters:
    -----------
    filter_keys : bool - Wenn True, werden andere Keys aus Nachbarn gefiltert.
                         Dies zeigt die echte Key→Werte Beziehung ohne Key-Clustering-Effekt.
    max_fetch : int - Maximale Anzahl initialer Nachbarn beim Filtern (Standard: 1000).
                      Höhere Werte = mehr Nicht-Key-Nachbarn, aber langsamer.
    
    Returns:
    --------
    Dict mit aggregierten Metriken und Einzelergebnissen
    """
    # Wähle zufällige Keys aus
    available_keys = [f"key_{i}" for i in range(len(df_with_keys))]
    
    # Filtere Keys die im Modell vorhanden sind
    valid_keys = [k for k in available_keys if k in model.wv]
    
    if len(valid_keys) == 0:
        return {'error': 'Keine gültigen Keys im Modell gefunden'}
    
    # Wähle Stichprobe
    sample_size = min(num_keys, len(valid_keys))
    sampled_keys = np.random.choice(valid_keys, size=sample_size, replace=False)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"  WORD2VEC KEY-TO-VALUES PROXIMITY EVALUATION")
        print(f"{'='*70}")
        print(f"Evaluiere {sample_size} Keys mit Top-{topn} Nicht-Key-Nachbarn")
        print(f"Filter Keys aus Nachbarn: {'Ja ✓' if filter_keys else 'Nein ✗'}")
        if filter_keys:
            print(f"Max Fetch pro Key: {max_fetch}")
        print(f"Hypothese: Keys sollten nahe bei ihren eigenen Zeilenwerten liegen")
        print(f"{'='*70}")
    
    results = []
    for key in sampled_keys:
        result = calculate_key_to_values_score(model, key, df_with_keys, topn, filter_keys, max_fetch, verbose)
        
        if 'error' not in result:
            results.append(result)
    
    if len(results) == 0:
        return {'error': 'Keine auswertbaren Keys gefunden'}
    
    # Aggregiere Ergebnisse
    avg_proximity = np.mean([r['proximity_score'] for r in results])
    avg_weighted_proximity = np.mean([r['weighted_proximity_score'] for r in results])
    total_found = sum([r['found_values'] for r in results])
    total_possible = sum([r['total_values_in_vocab'] for r in results])
    
    summary = {
        'model_score': avg_weighted_proximity,  # Hauptmetrik für Vergleiche
        'avg_proximity_score': avg_proximity,
        'avg_weighted_proximity_score': avg_weighted_proximity,
        'total_found_values': total_found,
        'total_possible_values': total_possible,
        'overall_recall': total_found / total_possible if total_possible > 0 else 0,
        'keys_evaluated': len(results),
        'topn': topn,
        'filter_keys': filter_keys,
        'max_fetch': max_fetch if filter_keys else None,
        'interpretation': interpret_proximity_score(avg_weighted_proximity),
        #'individual_results': results
    }
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"  ZUSAMMENFASSUNG")
        print(f"{'='*70}")
        print(f"Filter Keys aktiviert:       {'Ja ✓' if filter_keys else 'Nein ✗'}")
        if filter_keys:
            print(f"Max Fetch verwendet:         {max_fetch}")
        print(f"Model Score (gewichtet):     {avg_weighted_proximity:.3f}")
        print(f"Ø Proximity Score:           {avg_proximity:.3f}")
        print(f"Gesamt Recall:               {summary['overall_recall']:.3f} ({total_found}/{total_possible})")
        print(f"Keys evaluiert:              {len(results)}")
        print(f"Interpretation:              {summary['interpretation']}")
        print(f"{'='*70}\n")
    
    return summary

def interpret_proximity_score(score: float) -> str:
    """Interpretiert den Proximity Score."""
    if score >= 0.7:
        return "Exzellent - Keys kodieren ihre Zeilenwerte sehr gut"
    elif score >= 0.5:
        return "Gut - Keys haben starke Nähe zu ihren Zeilenwerten"
    elif score >= 0.3:
        return "Akzeptabel - Moderate Proximity zu Zeilenwerten"
    elif score >= 0.15:
        return "Schwach - Geringe Nähe zu eigenen Zeilenwerten"
    else:
        return "Ungenügend - Keys kodieren ihre Zeilenwerte kaum"

# ---------------------------------------------------
# Zusätzliche Analyse: Key-Spezifität
# ---------------------------------------------------

def analyze_key_specificity(model, key: str, df_with_keys, topn: int = 50) -> Dict:
    """
    Analysiert wie spezifisch ein Key seine Zeile repräsentiert.
    
    Misst das Verhältnis von:
    - Eigenen Zeilenwerten in Nachbarn (gut)
    - Fremden Zeilenwerten in Nachbarn (schlecht - zu generisch)
    - Sonstigen Wörtern in Nachbarn (neutral)
    """
    key_index = int(key.replace("key_", ""))
    own_values = set(get_key_values_new(key_index, filter_keys=True, df_with_keys=df_with_keys))
    
    # Sammle alle Zeilenwerte aus dem gesamten DataFrame
    all_row_values = set()
    for idx in range(len(df_with_keys)):
        row_values = get_key_values_new(idx, filter_keys=True, df_with_keys=df_with_keys)
        all_row_values.update(row_values)
    
    foreign_values = all_row_values - own_values
    
    try:
        neighbors = model.wv.most_similar(key, topn=topn)
    except KeyError:
        return {'error': f'{key} nicht im Vokabular'}
    
    own_count = 0
    foreign_count = 0
    other_count = 0
    
    for word, sim in neighbors:
        if word in own_values:
            own_count += 1
        elif word in foreign_values:
            foreign_count += 1
        else:
            other_count += 1
    
    specificity_score = (own_count - foreign_count) / topn if topn > 0 else 0
    
    return {
        'key': key,
        'own_values': own_count,
        'foreign_values': foreign_count,
        'other_tokens': other_count,
        'specificity_score': specificity_score,
        'interpretation': 'Spezifisch' if specificity_score > 0.2 else 'Generisch' if specificity_score < -0.2 else 'Neutral'
    }

# ---------------------------------------------------
# Modellvergleich
# ---------------------------------------------------

def compare_models(models_dict: Dict, df_with_keys, num_keys: int = 10, topn: int = 20, 
                   filter_keys: bool = True, max_fetch: int = 1000) -> pd.DataFrame:
    """
    Vergleicht mehrere Word2Vec-Modelle anhand ihrer Key-Quality.
    
    Parameters:
    -----------
    models_dict : Dict[str, Word2Vec]
        Dictionary mit Modellnamen als Keys und Modellen als Values
    
    Returns:
    --------
    pd.DataFrame mit Vergleichsergebnissen, sortiert nach Score
    """
    comparison_results = []
    
    for model_name, model in models_dict.items():
        print(f"\nEvaluiere Modell: {model_name}")
        result = evaluate_model_key_quality(model, df_with_keys, num_keys, topn, filter_keys, max_fetch, verbose=False)
        
        if 'error' not in result:
            comparison_results.append({
                'Model': model_name,
                'Quality Score': result['avg_weighted_proximity_score'],
                'Proximity Score': result['avg_proximity_score'],
                'Overall Recall': result['overall_recall'],
                'Values Found': f"{result['total_found_values']}/{result['total_possible_values']}",
                'Interpretation': result['interpretation']
            })
    
    if len(comparison_results) == 0:
        return pd.DataFrame()
    
    df_comparison = pd.DataFrame(comparison_results)
    df_comparison = df_comparison.sort_values('Quality Score', ascending=False)
    
    return df_comparison

