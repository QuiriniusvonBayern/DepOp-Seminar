from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
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

def neighborhood_discriminability_test(model, k=10, verbose=False):

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
    

    if verbose:
        print(f"Model {model}: NDR = {score:.4f}")

    return score



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



def feature_coherence_test(model_list, verbose=False, k=10, df_with_keys=None):
    vectors, rows = get_vectors_and_rows(model_list, df_with_keys)

    sim_matrix = cosine_similarity(vectors)
    np.fill_diagonal(sim_matrix, -1)

    feature_matrix = MultiLabelBinarizer().fit_transform(rows)

    scores = []
    for i in range(len(vectors)):
        neighbors = np.argsort(sim_matrix[i])[::-1][:k]
        scores.append(
            compute_feature_overlap_fast(i, neighbors, feature_matrix)
        )

    fc_score = float(np.mean(scores))


    if verbose:
        print(f"Feature Coherence (k={k}): {fc_score:.4f}")
    return fc_score


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

