from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# basic_vector_tests.py
# ---------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------
def interpret_similarity(token_pair, value):
    t1, t2 = token_pair
    expected_ranges = {
        ("Gender_Female", "Gender_Male"): (0.30, 0.90),
        ("Credit_Score_Fair", "Credit_Score_Good"): (0.25, 0.85),
        ("Churn_Yes", "Churn_No"): (-0.20, 0.30),
        ("Country_France", "Country_Germany"): (0.20, 0.80),
        ("Country_France", "Country_Spain"): (0.20, 0.80),
    }

    if (t1, t2) not in expected_ranges:
        return "Keine Referenz", False

    low, high = expected_ranges[(t1, t2)]
    if value < low: return "Zu niedrig", False
    if value > high: return "Zu hoch", False
    return "OK", True

def interpret_neighbors(token, neighbors, test, test_range):
    # Gibt nun direkt die Anzahl der korrekten Matches zurück
    if test != "none" and token.startswith(f"{test}_"):
        expected_prefix = f"{test}_"
        subset = neighbors[:min(test_range, len(neighbors))]
        
        matches = sum(1 for w, _ in subset if w.startswith(expected_prefix))
        total_checked = len(subset)
        
        return matches, total_checked
    return 0, 0

# ---------------------------------------------------
# Semantische Ähnlichkeit prüfen
# ---------------------------------------------------
def check_token_similarity(model, token_pairs, verbose):
    results = []
    
    for t1, t2 in token_pairs:
        try:
            sim = float(model.wv.similarity(t1, t2))
            msg, passed = interpret_similarity((t1, t2), sim)
            
            results.append({
                "pair": f"{t1}-{t2}",
                "similarity": sim,
                "passed": passed,
                "msg": msg
            })
            
            if verbose:
                print(f"{t1} ↔ {t2} = {sim:.4f} ({msg})")

        except KeyError:
            if verbose: print(f"{t1} oder {t2} fehlen.")
            results.append({
                "pair": f"{t1}-{t2}",
                "similarity": None,
                "passed": False,
                "msg": "Missing Token"
            })

    return results

# ---------------------------------------------------
# Nachbaranalyse
# ---------------------------------------------------
def check_neighbors(model, tokens, test, test_range, verbose, topn=10):
    results = []
    
    for t in tokens:
        try:
            nn = model.wv.most_similar(t, topn=topn)
            matches, checked = interpret_neighbors(t, nn, test, test_range)
            
            # Wir speichern die Quote (z.B. 1.0 = 100% korrekt)
            score = matches / checked if checked > 0 else 0
            
            results.append({
                "token": t,
                "category": test,
                "matches": matches,
                "checked_range": checked,
                "score": score
            })
            
            if verbose:
                print(f"{t}: {matches}/{checked} Nachbarn korrekt.")

        except KeyError:
            if verbose: print(f"{t} nicht gefunden.")
    
    return results

# ---------------------------------------------------
# Gesamter Qualitätscheck
# ---------------------------------------------------
def run_full_quality_check(model_list, verbose):
    if verbose:
        print("\n################################################")
        print("        AUTOMATISCHER W2V QUALITY CHECK")
        print("               Model Parameters:       ")
        print(f"Vektorraumgröße = {model_list[1]['Vector Size']} Fenstergröße  = {model_list[1]['Window']} Algorithmus = {model_list[1]['Algorithmus']}")
        print(f"         Epochen = {model_list[1]['Epochs']} Anzahl der Sätze = {model_list[1]['Satzanzahl']} Länge der Sätze = {model_list[1]['Satzlänge']}")
        print("################################################\n")

    if verbose:
        print("Running Basic Quality Checks...")

    model = model_list[0]
    # Wir sammeln alles in einem Dictionary
    data = {
        "semantic_similarity": [],
        "neighbors": []
    }

    # 1. Semantische Erwartungspaare
    data["semantic_similarity"] = check_token_similarity(model, [
        ("Gender_Female", "Gender_Male"),
        ("Credit_Score_Fair", "Credit_Score_Good"),
        ("Churn_Yes", "Churn_No"),
        ("Country_France", "Country_Germany"),
        ("Country_France", "Country_Spain"),
    ], verbose=verbose)

    # 2. Nachbar-Tests (Sammeln in einer Liste)
    neighbor_configs = [
        (["Country_France", "Country_Germany", "Country_Spain"], "Country", 2),
        (["Age_Young_Adults", "Age_Adults_their_Prime", "Age_Middle_aged"], "Age", 4),
        (["Salary_Very_low", "Salary_Below_average", "Salary_Very_high"], "Salary", 6),
        (["Credit_Score_Poor", "Credit_Score_Good", "Credit_Score_Excellent"], "Credit_Score", 4),
        (["Churn_Yes", "Churn_No"], "Churn", 1)
    ]

    for tokens, category, rng in neighbor_configs:
        res = check_neighbors(model, tokens, test=category, test_range=rng, verbose=verbose)
        data["neighbors"].extend(res)

    return data


#New Tests

def neighborhood_discriminability_test(model, k=10, verbose=False):
    results = []

    semantic_labels = get_semantic_labels()

    
    tokens, vectors, labels = extract_vectors_from_model(model[0], semantic_labels)

    sim_matrix = cosine_similarity(vectors)
    np.fill_diagonal(sim_matrix, -1.0)  # Selbstähnlichkeit entfernen

    intra_distances = []
    inter_distances = []

    for i in range(len(tokens)):
        neighbor_ids = np.argsort(sim_matrix[i])[-k:]

        for j in neighbor_ids:
            distance = 1 - sim_matrix[i, j]
            if labels[i] == labels[j]:
                intra_distances.append(distance)
            else:
                inter_distances.append(distance)

    score = np.mean(inter_distances) / np.mean(intra_distances)
    results.append(score)

    if verbose:
        print(f"Model {model}: NDR = {score:.4f}")

    return results



def extract_vectors_from_model(model, semantic_labels):
    tokens = []
    vectors = []
    labels = []

    for token, label in semantic_labels.items():
        if token in model.wv:
            tokens.append(token)
            vectors.append(model.wv[token])
            labels.append(label)

    return tokens, np.vstack(vectors), labels





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

        base = neigh_sets[0]
        for other in neigh_sets[1:]:
            jaccards.append(jaccard_index(base, other))

    rs_score = float(np.mean(jaccards))

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






def feature_coherence_test(model_list, verbose=False, k=10):
    vectors, rows = get_vectors_and_rows(model_list)
    scores = []

    for i, v in enumerate(vectors):
        neighbors = top_k_neighbors(v, vectors, k)
        neighbor_rows = [rows[j] for j in neighbors]

        coherence = compute_feature_overlap(rows[i], neighbor_rows)
        scores.append(coherence)

    fc_score = float(np.mean(scores))

    if verbose:
        print(f"Feature Coherence (k={k}): {fc_score:.4f}")

    return fc_score


def get_vectors_and_rows(model_list):
    """
    Extrahiert:
    - vectors: np.ndarray shape (N, D)
    - rows:    List[set] mit diskreten Feature-Tokens
    """

    vectors = []
    rows = []

    for seed, vectors in model_list["data"]:
        print(seed, vectors)

    # model_list["data"] = [(seed, list_of_rows), ...]
    for seed, row_entries in model_list["data"]:
        for row in row_entries:
            vectors.append(row["vector"])
            rows.append(row["features"])

    return np.array(vectors), rows


def top_k_neighbors(v, vectors, k):
    sims = cosine_similarity(v.reshape(1, -1), vectors)[0]
    sims_idx = np.argsort(sims)[::-1]

    # Entferne Self-Match (Index 0)
    neighbors = [idx for idx in sims_idx if not np.allclose(v, vectors[idx])]

    return neighbors[:k]

def compute_feature_overlap(row_features, neighbor_rows):
    overlaps = []

    for n_features in neighbor_rows:
        intersection = len(row_features & n_features)
        union = len(row_features | n_features)

        if union == 0:
            overlaps.append(0.0)
        else:
            overlaps.append(intersection / union)

    return np.mean(overlaps)
