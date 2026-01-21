from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.metrics import silhouette_score
import numpy as np

def get_key_values(key, filter, prefix="key", df_with_keys=None):
    values_with_keys = df_with_keys.iloc[key] 
    
    result = []
    for i, value in enumerate(values_with_keys):
        column_name = df_with_keys.columns[i]  # Get actual column name
        
        # Apply filter if needed
        if filter and str(value).startswith(prefix):
            continue
        
        # Format with actual column name
        result.append(f'"{column_name}={value}"')
    
    return result

def get_key_values_by_column(key, filter, prefix="key", df_with_keys=None, include_cols=None):
    row = df_with_keys.iloc[key]
    result = []

    for col, value in row.items():
        if include_cols is not None and col not in include_cols:
            continue
        if filter and str(value).startswith(prefix):
            continue
        result.append(f"{col}={value}")
    return result


def extract_vectors_from_model(model, semantic_labels):
    """Extrahiert Vektoren und wandelt Labels in numerische Werte um."""
    tokens = []
    vectors = []
    categorical_labels = []  # Kategorische Labels sammeln
    
    for token, label in semantic_labels.items():
        if token in model.wv:
            tokens.append(token)
            vectors.append(model.wv[token])
            categorical_labels.append(label)
    
    le = None

    if categorical_labels:
        le = LabelEncoder()
        numeric_labels = le.fit_transform(categorical_labels)
    else:
        numeric_labels = np.array([])
    
    return tokens, np.vstack(vectors) if vectors else np.array([]), numeric_labels, le

def neighborhood_discriminability_test_silhouette(model, verbose=False):
    """
    Ersetzt die alte ND-Ratio durch den Silhouette-Koeffizienten.
    Prüft, wie gut die Vektoren innerhalb ihrer semantischen Klassen 
    gruppiert sind im Vergleich zu anderen Klassen.
    """
    # ACHTUNG: Ihr Code verwendet model[0] - ist model eine Liste/Tupel?
    # Wenn model das Word2Vec-Modell direkt ist, verwenden Sie es direkt:
    if isinstance(model, (list, tuple)):
        wv_model = model[0]  # Wenn model eine Liste/Tupel ist
    else:
        wv_model = model  # Wenn model direkt das Modell ist
    
    # 1. Labels und Vektoren extrahieren
    semantic_labels = get_semantic_labels()
    
    # Jetzt mit Label-Kodierung
    tokens, vectors, numeric_labels, label_encoder = extract_vectors_from_model(
        wv_model, semantic_labels
    )
    
    # Überprüfen, ob wir genug Daten haben
    if len(vectors) == 0:
        if verbose: 
            print("ND Warnung: Keine Vektoren gefunden.")
        return 0.0
    
    if len(np.unique(numeric_labels)) < 2:
        if verbose: 
            print(f"ND Warnung: Nur {len(np.unique(numeric_labels))} Klasse(n) gefunden. "
                  f"Silhouette benötigt mindestens 2 Klassen.")
        return 0.0
    
    if len(numeric_labels) < 3:
        if verbose:
            print(f"ND Warnung: Nur {len(numeric_labels)} Samples gefunden. "
                  f"Silhouette benötigt mindestens 3 Samples.")
        return 0.0
    
    # 2. Silhouette Score berechnen
    try:
        # Verwende 'cosine' als Distanzmaß
        score = silhouette_score(vectors, numeric_labels, metric='cosine')
        
        if verbose:
            print(f"Neighborhood Discriminability (Silhouette): {score:.4f}")
            print(f"Anzahl Samples: {len(vectors)}")
            print(f"Anzahl Klassen: {len(np.unique(numeric_labels))}")
            print(f"Klassen: {label_encoder.classes_}")
            
            # Zusätzliche Diagnose-Informationen
            unique, counts = np.unique(numeric_labels, return_counts=True)
            class_counts = {label_encoder.inverse_transform([u])[0]: int(c) for u, c in zip(unique, counts)}
            print(f"Klassenverteilung: {class_counts}")

        
        # Silhouette Score liegt zwischen -1 und 1
        # Negative Scores sind schlecht, also normalisieren wir auf [0, 1]
        nd_norm = max(0, score)
        
        return float(nd_norm)
        
    except Exception as e:
        if verbose:
            print(f"Fehler bei Silhouette-Berechnung: {e}")
        return 0.0



# Hilfsfunktion zur Überprüfung Ihres Models
def check_model_vocabulary(model):
    """Überprüft, welche Wörter aus Ihren Labels im Modell vorhanden sind."""
    if isinstance(model, (list, tuple)):
        wv_model = model[0]
    else:
        wv_model = model
    
    semantic_labels = get_semantic_labels()
    
    print("VOKABULAR-ÜBERPRÜFUNG")
    print("=" * 60)
    
    # Prüfe jedes Label-Wort
    found = []
    missing = []
    
    for token in semantic_labels.keys():
        if token in wv_model.wv:
            found.append(token)
        else:
            missing.append(token)
    
    print(f"Gefunden: {len(found)}/{len(semantic_labels)} Wörtern")
    print(f"Fehlend: {len(missing)}/{len(semantic_labels)} Wörtern")
    
    if missing:
        print("\nFehlende Wörter (erste 10):")
        for token in missing[:10]:
            label = semantic_labels[token]
            print(f"  {token} -> {label}")
    
    return found, missing

def get_semantic_labels():
    return {
        # Countries
        "Country_Spain": "Country",
        "Country_Germany": "Country",
        "Country_France": "Country",
        
        # Gender
        "Gender_Male": "Gender",
        "Gender_Female": "Gender",
        
        # Age Categories
        "Age_Young_Adults": "Age",
        "Age_Adults_in_their_Prime": "Age",
        "Age_Middle_aged": "Age",
        "Age_Pre_retirees": "Age",
        "Age_Young_Seniors": "Age",
        
        # Tenure (dynamisch generiert, daher allgemeine Form)
        "Tenure_0": "Tenure",
        "Tenure_1": "Tenure",
        "Tenure_2": "Tenure",
        "Tenure_3": "Tenure",
        "Tenure_4": "Tenure",
        "Tenure_5": "Tenure",
        "Tenure_6": "Tenure",
        "Tenure_7": "Tenure",
        "Tenure_8": "Tenure",
        "Tenure_9": "Tenure",
        "Tenure_10": "Tenure",
        
        # Balance Clusters
        "Balance_Cluster_1": "Balance",
        "Balance_Cluster_2": "Balance",
        "Balance_Cluster_3": "Balance",
        "Balance_Cluster_4": "Balance",
        "Balance_Cluster_5": "Balance",
        "Balance_Cluster_6": "Balance",
        "Balance_Cluster_7": "Balance",
        "Balance_Cluster_8": "Balance",
        "Balance_Cluster_9": "Balance",
        "Balance_Cluster_10": "Balance",
        
        # Products Number
        "ProductsNumber_1": "ProductsNumber",
        "ProductsNumber_2": "ProductsNumber",
        "ProductsNumber_3": "ProductsNumber",
        "ProductsNumber_4": "ProductsNumber",
        
        # Credit Card
        "CreditCard_Yes": "CreditCard",
        "CreditCard_No": "CreditCard",
        
        # Active Member
        "ActiveMember_Yes": "ActiveMember",
        "ActiveMember_No": "ActiveMember",
        
        # Salary Categories
        "Salary_Very_low": "Salary",
        "Salary_Low": "Salary",
        "Salary_Below_average": "Salary",
        "Salary_Average": "Salary",
        "Salary_Above_average": "Salary",
        "Salary_High": "Salary",
        "Salary_Very_high": "Salary",
        
        # Credit Score Categories
        "Credit_Score_Poor": "CreditScore",
        "Credit_Score_Fair": "CreditScore",
        "Credit_Score_Good": "CreditScore",
        "Credit_Score_Very_Good": "CreditScore",
        "Credit_Score_Excellent": "CreditScore",
        
        # Churn
        "Churn_Yes": "Churn",
        "Churn_No": "Churn",
    }


def rank_stability_test_average(model_list, verbose=False, max_anchors=100):
    seed_runs = model_list["data"]

    if len(seed_runs) < 2:
        raise ValueError("Rank Stability benötigt mindestens zwei Seeds")

    normalized_vectors = []
    for seed, vectors in seed_runs:
        # Konvertiere Series von Listen zu 2D-NumPy-Array
        vectors_array = np.array(vectors["vector"].tolist())
        # Normalisiere
        vectors_normalized = vectors_array / np.linalg.norm(vectors_array, axis=1, keepdims=True)
        normalized_vectors.append((seed, vectors_normalized))

    seed_runs = normalized_vectors

    if verbose:
        print(seed for seed, _ in seed_runs)
        print(vectors for vectors in seed_runs)

    anchor_indices = select_anchor_vectors(seed_runs, max_anchors)

    rs_5 = rank_stability_test(anchor_indices, seed_runs, k=5, verbose=verbose)
    rs_10 = rank_stability_test(anchor_indices, seed_runs, k=10, verbose=verbose)
    rs_20 = rank_stability_test(anchor_indices, seed_runs, k=20, verbose=verbose)

    average_result = np.mean([
                            rs_5,
                            rs_10,
                            rs_20
                        ])

    return average_result, rs_5, rs_10, rs_20

def rank_stability_test(anchor_indices, seed_runs, k, verbose=False):
    jaccards = []
    
    for anchor_idx in anchor_indices:
        neigh_sets = []
        
        for seed, vectors in seed_runs:
            neigh = top_k_neighbors_rs(anchor_idx, vectors, k)
            neigh_sets.append(set(neigh))
        
        # ALLE paarweisen Vergleiche durchführen
        for i in range(len(neigh_sets)):
            for j in range(i + 1, len(neigh_sets)):
                jaccards.append(jaccard_index(neigh_sets[i], neigh_sets[j]))
    
    rs_score = float(np.mean(jaccards)) if jaccards else 0.0
    
    if verbose:
        print(f"Rank Stability (k={k}): {rs_score:.4f}")
    
    return rs_score

def select_anchor_vectors(seed_runs, max_anchors=100):
    _, vectors = seed_runs[0]
    N = vectors.shape[0]
    idx = np.random.choice(N, min(max_anchors, N), replace=False)
    return idx

def top_k_neighbors_rs(anchor_idx, vectors, k):
    anchor = vectors[anchor_idx]
    sims = vectors @ anchor
    order = np.argsort(sims)[::-1]
    order = order[order != anchor_idx]  # sich selbst entfernen
    return order[:k]


def jaccard_index(a, b):
    return len(a & b) / len(a | b)



def get_vectors_and_rows_for_seed(model_list, df_with_keys, seed_target):
    sentence_vectors, rows = [], []
    for seed, vectors in model_list["data"]:
        if seed == seed_target:
            for index, vector in zip(vectors["index"], vectors["vector"]):
                # Prüfen, ob Zeile index wirklich key_{index+1} enthält
                row = df_with_keys.iloc[index].astype(str).tolist()
                assert any(v == f"key_{index+1}" for v in row), f"Mismatch at index={index}"
                sentence_vectors.append(vector)
                fc_cols = ["credit_card", "gender", "balance", "credit_score", "churn"]

                rows.append(get_key_values_by_column(index, True, prefix="key", df_with_keys=df_with_keys, include_cols=fc_cols))

    return np.array(sentence_vectors), rows

def feature_coherence_test(
    model_list,
    verbose=False,
    k=10,
    df_with_keys=None,
    normalize=True,
    clip_normalized=False,
    rng_seed=42,
    return_details=False
):
    """
    Raw FC:
      FC_global = mean_seed mean_r mean_{n in N_k(r)} Jaccard(F(r), F(n))

    Random Baseline:
      FC_random = mean_seed mean_r mean_{n in Random_k(r)} Jaccard(F(r), F(n))

    Normalized (Lift):
      FC_norm = (FC_knn - FC_random) / (1 - FC_random)

    Notes:
    - Raw FC und FC_random liegen in [0,1].
    - FC_norm kann negativ sein (schlechter als Zufall). Mit clip_normalized=True wird auf [0,1] gekappt.
    - Voraussetzung: vectors["index"] muss 0-basierte Positionsindizes in df_with_keys sein (passt zu df_with_keys.iloc[key]).
    """

    seed_fc_scores = []
    seed_rand_scores = []
    #verbose=True

    for seed, _ in model_list["data"]:
        vectors, rows = get_vectors_and_rows_for_seed(model_list, df_with_keys, seed)
        n = len(vectors)

        # Für random neighbors ohne replacement braucht man mindestens k+1 Instanzen
        if n < k + 1:
            continue

        # Similarity-Matrix für KNN
        sim_matrix = cosine_similarity(vectors)
        np.fill_diagonal(sim_matrix, -1)

        # Feature-Matrix (binarisiert)
        feature_matrix = MultiLabelBinarizer().fit_transform(rows)

        # Reproduzierbarer RNG pro Seed
        rng = np.random.default_rng(int(rng_seed) + int(seed))

        all_indices = np.arange(n)
        fc_per_instance = []
        rand_per_instance = []

        for i in range(n):
            # --- KNN neighbors ---
            neighbors = np.argsort(sim_matrix[i])[::-1][:k]
            fc_per_instance.append(compute_feature_overlap_fast(i, neighbors, feature_matrix))

            # --- Random baseline neighbors ---
            candidates = all_indices[all_indices != i]
            rand_neighbors = rng.choice(candidates, size=k, replace=False)
            rand_per_instance.append(compute_feature_overlap_fast(i, rand_neighbors, feature_matrix))

        seed_fc_scores.append(float(np.mean(fc_per_instance)))
        seed_rand_scores.append(float(np.mean(rand_per_instance)))

    fc_knn = float(np.mean(seed_fc_scores)) if seed_fc_scores else 0.0
    fc_random = float(np.mean(seed_rand_scores)) if seed_rand_scores else 0.0

    if not normalize:
        if verbose:
            print(f"Feature Coherence raw (k={k}, seeds={len(seed_fc_scores)}): {fc_knn:.4f}")
            print(f"Feature Coherence random baseline: {fc_random:.4f}")
        return (fc_knn, fc_random) if return_details else fc_knn

    denom = 1.0 - fc_random
    if denom <= 0:
        # Extremfall: Baseline == 1.0, dann ist Normalisierung nicht sinnvoll.
        fc_norm = 0.0
    else:
        fc_norm = (fc_knn - fc_random) / denom

    if clip_normalized:
        fc_norm = float(np.clip(fc_norm, 0.0, 1.0))

    if verbose:
        print(f"Feature Coherence raw (k={k}, seeds={len(seed_fc_scores)}): {fc_knn:.4f}")
        print(f"Feature Coherence random baseline: {fc_random:.4f}")
        print(f"Feature Coherence normalized: {fc_norm:.4f}")

    return (float(fc_norm), fc_knn, fc_random) if return_details else float(fc_norm)




def get_vectors_and_rows(model_list, df_with_keys=None):
    """
    Extrahiert:
    - vectors: np.ndarray shape (N, D)
    - rows:    List[set] mit diskreten Feature-Tokens
    """

    sentence_vectors = []
    rows = []

    for seed, vectors in model_list["data"]:
        if seed == 1:
            for index, vector in zip(vectors["index"], vectors["vector"]):
                sentence_vectors.append(vector)
                rows.append(get_key_values(index, True, prefix="key", df_with_keys=df_with_keys))

    return np.array(sentence_vectors), rows


def top_k_neighbors(v, vectors, k):
    sims = cosine_similarity(v.reshape(1, -1), vectors)[0]
    sims_idx = np.argsort(sims)[::-1]

    # Entferne Self-Match (Index 0)
    neighbors = [idx for idx in sims_idx if not np.allclose(v, vectors[idx])]

    return neighbors[:k]

def compute_feature_overlap_fast(i, neighbors, feature_matrix):
    row_vec = feature_matrix[i]
    neighbor_vecs = feature_matrix[neighbors]

    intersection = np.logical_and(neighbor_vecs, row_vec).sum(axis=1)
    union = np.logical_or(neighbor_vecs, row_vec).sum(axis=1)

    overlap = np.divide(
        intersection, union,
        out=np.zeros_like(intersection, dtype=float),
        where=union > 0
    )
    return overlap.mean()

