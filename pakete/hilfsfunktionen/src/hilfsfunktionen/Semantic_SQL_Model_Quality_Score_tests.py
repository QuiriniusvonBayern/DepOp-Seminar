from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.metrics import silhouette_score, silhouette_samples
import numpy as np


def format_key_value_pairs(row_index, apply_filter, prefix="key", dataframe=None):
    """
    Retrieve and format column-value pairs from a specific row in a DataFrame.

    Parameters
    ----------
    row_index : int
        Index of the row to extract values from.
    apply_filter : bool
        Whether to filter out values starting with the specified prefix.
    prefix : str, optional
        Prefix used for filtering (default is "key").
    dataframe : pd.DataFrame, optional
        DataFrame containing the data.

    Returns
    -------
    list
        List of formatted strings "column_name=value".
    """
    row_values = dataframe.iloc[row_index]
    
    result = []
    for i, value in enumerate(row_values):
        column_name = dataframe.columns[i]
        
        if apply_filter and str(value).startswith(prefix):
            continue
        
        result.append(f'"{column_name}={value}"')
    
    return result


def format_key_value_pairs_by_column(row_index, apply_filter, prefix="key", dataframe=None, include_columns=None):
    """
    Retrieve and format column-value pairs from specific columns of a row.

    Parameters
    ----------
    row_index : int
        Index of the row to extract values from.
    apply_filter : bool
        Whether to filter out values starting with the specified prefix.
    prefix : str, optional
        Prefix used for filtering (default is "key").
    dataframe : pd.DataFrame, optional
        DataFrame containing the data.
    include_columns : list, optional
        List of column names to include. If None, all columns are used.

    Returns
    -------
    list
        List of formatted strings "column_name=value".
    """
    row = dataframe.iloc[row_index]
    result = []

    for column, value in row.items():
        if include_columns is not None and column not in include_columns:
            continue
        if apply_filter and str(value).startswith(prefix):
            continue
        result.append(f"{column}={value}")
    return result


def extract_vectors_and_labels_from_model(model, semantic_labels):
    """
    Extract vectors from a word2vec model and convert semantic labels to numeric values.

    Parameters
    ----------
    model : gensim.models.Word2Vec or tuple
        Word2Vec model or a tuple containing the model as first element.
    semantic_labels : dict
        Dictionary mapping tokens to their semantic category labels.

    Returns
    -------
    tuple
        (tokens, vectors, numeric_labels, label_encoder)
    """
    if isinstance(model, (list, tuple)):
        word2vec_model = model[0]
    else:
        word2vec_model = model
    
    tokens = []
    vectors = []
    categorical_labels = []
    
    for token, label in semantic_labels.items():
        if token in word2vec_model.wv:
            tokens.append(token)
            vectors.append(word2vec_model.wv[token])
            categorical_labels.append(label)
    
    label_encoder = None
    if categorical_labels:
        label_encoder = LabelEncoder()
        numeric_labels = label_encoder.fit_transform(categorical_labels)
    else:
        numeric_labels = np.array([])
    
    return tokens, np.vstack(vectors) if vectors else np.array([]), numeric_labels, label_encoder


def neighborhood_discriminability_silhouette(model, verbose=False):
    """
    Calculate neighborhood discriminability using the silhouette coefficient.

    Measures how well vectors are grouped within their semantic classes
    compared to other classes. Values are clipped at zero per sample
    before averaging.

    Parameters
    ----------
    model : gensim.models.Word2Vec or tuple
        Word2Vec model or a tuple containing the model.
    verbose : bool, optional
        If True, print detailed information.

    Returns
    -------
    float
        Mean of positive silhouette scores across all samples.
    """
    if isinstance(model, (list, tuple)):
        word2vec_model = model[0]
    else:
        word2vec_model = model
    
    semantic_labels = get_semantic_labels()
    
    tokens, vectors, numeric_labels, label_encoder = extract_vectors_and_labels_from_model(
        word2vec_model, semantic_labels
    )
    
    if len(vectors) == 0:
        if verbose:
            print("ND Warning: No vectors found.")
        return 0.0
    
    if len(np.unique(numeric_labels)) < 2:
        if verbose:
            print(f"ND Warning: Only {len(np.unique(numeric_labels))} class(es) found. "
                  f"Silhouette requires at least 2 classes.")
        return 0.0
    
    if len(numeric_labels) < 3:
        if verbose:
            print(f"ND Warning: Only {len(numeric_labels)} samples found. "
                  f"Silhouette requires at least 3 samples.")
        return 0.0
    
    try:
        sample_silhouette = silhouette_samples(vectors, numeric_labels, metric='cosine')
        nd_score = float(np.maximum(0.0, sample_silhouette).mean())
        
        if verbose:
            print(f"Neighborhood Discriminability (Silhouette, per-sample clipped): {nd_score:.4f}")
            print(f"Number of samples: {len(vectors)}")
            print(f"Number of classes: {len(np.unique(numeric_labels))}")
            print(f"Classes: {label_encoder.classes_}")
            
            unique, counts = np.unique(numeric_labels, return_counts=True)
            class_counts = {label_encoder.inverse_transform([u])[0]: int(c) for u, c in zip(unique, counts)}
            print(f"Class distribution: {class_counts}")
        
        return nd_score
        
    except Exception as e:
        if verbose:
            print(f"Error in silhouette calculation: {e}")
        return 0.0


def check_model_vocabulary_coverage(model):
    """
    Check which tokens from semantic labels are present in the model vocabulary.

    Parameters
    ----------
    model : gensim.models.Word2Vec or tuple
        Word2Vec model or a tuple containing the model.

    Returns
    -------
    tuple
        (found_tokens, missing_tokens)
    """
    if isinstance(model, (list, tuple)):
        word2vec_model = model[0]
    else:
        word2vec_model = model
    
    semantic_labels = get_semantic_labels()
    
    print("VOCABULARY CHECK")
    print("=" * 60)
    
    found_tokens = []
    missing_tokens = []
    
    for token in semantic_labels.keys():
        if token in word2vec_model.wv:
            found_tokens.append(token)
        else:
            missing_tokens.append(token)
    
    print(f"Found: {len(found_tokens)}/{len(semantic_labels)} tokens")
    print(f"Missing: {len(missing_tokens)}/{len(semantic_labels)} tokens")
    
    if missing_tokens:
        print("\nMissing tokens (first 10):")
        for token in missing_tokens[:10]:
            label = semantic_labels[token]
            print(f"  {token} -> {label}")
    
    return found_tokens, missing_tokens


def get_semantic_labels():
    """
    Return a dictionary mapping token strings to their semantic category labels.

    Returns
    -------
    dict
        Mapping from token to category.
    """
    return {
        "Country_Spain": "Country",
        "Country_Germany": "Country",
        "Country_France": "Country",
        
        "Gender_Male": "Gender",
        "Gender_Female": "Gender",
        
        "Age_Young_Adults": "Age",
        "Age_Adults_in_their_Prime": "Age",
        "Age_Middle_aged": "Age",
        "Age_Pre_retirees": "Age",
        "Age_Young_Seniors": "Age",
        
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
        
        "ProductsNumber_1": "ProductsNumber",
        "ProductsNumber_2": "ProductsNumber",
        "ProductsNumber_3": "ProductsNumber",
        "ProductsNumber_4": "ProductsNumber",
        
        "CreditCard_Yes": "CreditCard",
        "CreditCard_No": "CreditCard",
        
        "ActiveMember_Yes": "ActiveMember",
        "ActiveMember_No": "ActiveMember",
        
        "Salary_Very_low": "Salary",
        "Salary_Low": "Salary",
        "Salary_Below_average": "Salary",
        "Salary_Average": "Salary",
        "Salary_Above_average": "Salary",
        "Salary_High": "Salary",
        "Salary_Very_high": "Salary",
        
        "Credit_Score_Poor": "CreditScore",
        "Credit_Score_Fair": "CreditScore",
        "Credit_Score_Good": "CreditScore",
        "Credit_Score_Very_Good": "CreditScore",
        "Credit_Score_Excellent": "CreditScore",
        
        "Churn_Yes": "Churn",
        "Churn_No": "Churn",
    }


def rank_stability_average(model_list, verbose=False, max_anchors=100):
    """
    Calculate average rank stability across multiple k values (5, 10, 20).

    Parameters
    ----------
    model_list : dict
        Dictionary containing model runs with key "data" mapping to list of (seed, vectors) tuples.
    verbose : bool, optional
        If True, print detailed information.
    max_anchors : int, optional
        Maximum number of anchor vectors to use.

    Returns
    -------
    tuple
        (average_score, score_k5, score_k10, score_k20)
    """
    seed_runs = model_list["data"]
    
    if len(seed_runs) < 2:
        raise ValueError("Rank stability requires at least two seeds")
    
    normalized_vectors = []
    for seed, vectors in seed_runs:
        vectors_array = np.array(vectors["vector"].tolist())
        vectors_normalized = vectors_array / np.linalg.norm(vectors_array, axis=1, keepdims=True)
        normalized_vectors.append((seed, vectors_normalized))
    
    seed_runs = normalized_vectors
    
    if verbose:
        print([seed for seed, _ in seed_runs])
        print([vectors for vectors in seed_runs])
    
    anchor_indices = select_anchor_vectors(seed_runs, max_anchors)
    
    score_k5 = calculate_rank_stability(anchor_indices, seed_runs, k=5, verbose=verbose)
    score_k10 = calculate_rank_stability(anchor_indices, seed_runs, k=10, verbose=verbose)
    score_k20 = calculate_rank_stability(anchor_indices, seed_runs, k=20, verbose=verbose)
    
    average_score = np.mean([score_k5, score_k10, score_k20])
    
    return average_score, score_k5, score_k10, score_k20


def calculate_rank_stability(anchor_indices, seed_runs, k, verbose=False):
    """
    Calculate rank stability for a specific k value using Jaccard similarity.

    Parameters
    ----------
    anchor_indices : array-like
        Indices of anchor vectors to use.
    seed_runs : list
        List of (seed, vectors) tuples.
    k : int
        Number of nearest neighbors to consider.
    verbose : bool, optional
        If True, print detailed information.

    Returns
    -------
    float
        Mean Jaccard similarity across all pairwise comparisons.
    """
    jaccard_scores = []
    
    for anchor_idx in anchor_indices:
        neighbor_sets = []
        
        for seed, vectors in seed_runs:
            neighbors = find_top_k_neighbors(anchor_idx, vectors, k)
            neighbor_sets.append(set(neighbors))
        
        for i in range(len(neighbor_sets)):
            for j in range(i + 1, len(neighbor_sets)):
                jaccard_scores.append(calculate_jaccard_index(neighbor_sets[i], neighbor_sets[j]))
    
    rs_score = float(np.mean(jaccard_scores)) if jaccard_scores else 0.0
    
    if verbose:
        print(f"Rank Stability (k={k}): {rs_score:.4f}")
    
    return rs_score


def select_anchor_vectors(seed_runs, max_anchors=100):
    """
    Randomly select anchor vectors from the first seed run.

    Parameters
    ----------
    seed_runs : list
        List of (seed, vectors) tuples.
    max_anchors : int, optional
        Maximum number of anchors to select.

    Returns
    -------
    ndarray
        Array of randomly selected indices.
    """
    _, vectors = seed_runs[0]
    num_vectors = vectors.shape[0]
    indices = np.random.choice(num_vectors, min(max_anchors, num_vectors), replace=False)
    return indices


def find_top_k_neighbors(anchor_idx, vectors, k):
    """
    Find the k nearest neighbors of an anchor vector (excluding itself).

    Parameters
    ----------
    anchor_idx : int
        Index of the anchor vector.
    vectors : ndarray
        Array of all vectors.
    k : int
        Number of neighbors to find.

    Returns
    -------
    list
        Indices of the k nearest neighbors.
    """
    anchor = vectors[anchor_idx]
    similarities = vectors @ anchor
    order = np.argsort(similarities)[::-1]
    order = order[order != anchor_idx]
    return order[:k]


def calculate_jaccard_index(set_a, set_b):
    """
    Calculate Jaccard similarity between two sets.

    Parameters
    ----------
    set_a : set
        First set.
    set_b : set
        Second set.

    Returns
    -------
    float
        Jaccard similarity (intersection over union).
    """
    return len(set_a & set_b) / len(set_a | set_b)


def extract_feature_coherence_set(dataframe, row_index, feature_columns):
    """
    Extract a set of discrete feature tokens from specific columns of a row.

    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame containing feature coherence labels.
    row_index : int
        Index of the row.
    feature_columns : list
        List of column names containing feature coherence labels.

    Returns
    -------
    set
        Set of feature tokens (e.g., {'FCRISK_High', 'FCVAL_Medium'}).
    """
    values = dataframe.iloc[row_index][feature_columns].astype(str).tolist()
    return set(v for v in values if v and v.lower() != "nan")


def get_vectors_and_feature_rows_for_seed(model_list, dataframe, target_seed, feature_columns):
    """
    Extract vectors and corresponding feature sets for a specific seed.

    Parameters
    ----------
    model_list : dict
        Dictionary containing model runs.
    dataframe : pd.DataFrame
        DataFrame containing feature coherence labels.
    target_seed : int
        Seed value to extract.
    feature_columns : list
        List of column names containing feature coherence labels.

    Returns
    -------
    tuple
        (vectors_array, feature_sets_list)
    """
    vectors_list = []
    feature_sets = []
    
    for seed, vectors in model_list["data"]:
        if seed == target_seed:
            for index, vector in zip(vectors["index"], vectors["vector"]):
                row = dataframe.iloc[index].astype(str).tolist()
                assert any(v == f"key_{index+1}" for v in row), f"Mismatch at index={index}"
                
                vectors_list.append(vector)
                feature_sets.append(extract_feature_coherence_set(dataframe, index, feature_columns))
    
    return np.array(vectors_list), feature_sets


def feature_coherence_test(
    model_list,
    dataframe_with_labels,
    verbose=False,
    k=10,
    dataframe_with_keys=None,
    normalize=True,
    clip_normalized=False,
    rng_seed=42,
    return_details=False,
    feature_columns=None,
):
    """
    Calculate feature coherence score measuring semantic consistency of nearest neighbors.

    Compares feature overlap among k-nearest neighbors against random neighbors.

    Parameters
    ----------
    model_list : dict
        Dictionary containing model runs.
    dataframe_with_labels : pd.DataFrame
        DataFrame containing feature coherence labels.
    verbose : bool, optional
        If True, print detailed information.
    k : int, optional
        Number of neighbors to consider.
    dataframe_with_keys : pd.DataFrame, optional
        DataFrame with key information (unused, kept for compatibility).
    normalize : bool, optional
        If True, normalize score against random baseline.
    clip_normalized : bool, optional
        If True, clip normalized score to [0, 1].
    rng_seed : int, optional
        Random seed for reproducibility.
    return_details : bool, optional
        If True, return raw scores as well.
    feature_columns : list, optional
        List of column names containing feature coherence labels.

    Returns
    -------
    float or tuple
        Normalized score, or tuple of (normalized, raw_knn, raw_random) if return_details=True.
    """
    if feature_columns is None:
        feature_columns = ["fc_risk_tier", "fc_value_tier", "fc_engagement_tier"]
    
    seed_fc_scores = []
    seed_random_scores = []
    
    for seed, _ in model_list["data"]:
        vectors, feature_sets = get_vectors_and_feature_rows_for_seed(
            model_list=model_list,
            dataframe=dataframe_with_labels,
            target_seed=seed,
            feature_columns=feature_columns
        )
        num_samples = len(vectors)
        
        if verbose:
            print("Feature sets sample:", feature_sets[0] if feature_sets else "No feature sets")
        
        if num_samples < k + 1:
            continue
        
        similarity_matrix = cosine_similarity(vectors)
        np.fill_diagonal(similarity_matrix, -1)
        
        feature_matrix = MultiLabelBinarizer().fit_transform(feature_sets)
        
        rng = np.random.default_rng(int(rng_seed) + int(seed))
        
        all_indices = np.arange(num_samples)
        fc_per_instance = []
        random_per_instance = []
        
        for i in range(num_samples):
            neighbors = np.argsort(similarity_matrix[i])[::-1][:k]
            fc_per_instance.append(compute_feature_overlap_fast(i, neighbors, feature_matrix))
            
            candidates = all_indices[all_indices != i]
            random_neighbors = rng.choice(candidates, size=k, replace=False)
            random_per_instance.append(compute_feature_overlap_fast(i, random_neighbors, feature_matrix))
        
        seed_fc_scores.append(float(np.mean(fc_per_instance)))
        seed_random_scores.append(float(np.mean(random_per_instance)))
    
    fc_knn = float(np.mean(seed_fc_scores)) if seed_fc_scores else 0.0
    fc_random = float(np.mean(seed_random_scores)) if seed_random_scores else 0.0
    
    if not normalize:
        if verbose:
            print(f"Feature Coherence raw (k={k}, seeds={len(seed_fc_scores)}): {fc_knn:.4f}")
            print(f"Feature Coherence random baseline: {fc_random:.4f}")
        return (fc_knn, fc_random) if return_details else fc_knn
    
    denominator = 1.0 - fc_random
    fc_normalized = 0.0 if denominator <= 0 else (fc_knn - fc_random) / denominator
    
    if clip_normalized:
        fc_normalized = float(np.clip(fc_normalized, 0.0, 1.0))
    
    if verbose:
        print(f"Feature Coherence raw (k={k}, seeds={len(seed_fc_scores)}): {fc_knn:.4f}")
        print(f"Feature Coherence random baseline: {fc_random:.4f}")
        print(f"Feature Coherence normalized: {fc_normalized:.4f}")
    
    return (float(fc_normalized), fc_knn, fc_random) if return_details else float(fc_normalized)


def get_vectors_and_feature_rows(model_list, dataframe_with_keys=None):
    """
    Extract vectors and corresponding feature sets for the first seed.

    Parameters
    ----------
    model_list : dict
        Dictionary containing model runs.
    dataframe_with_keys : pd.DataFrame, optional
        DataFrame containing key information.

    Returns
    -------
    tuple
        (vectors_array, feature_sets_list)
    """
    vectors_list = []
    feature_sets = []
    
    for seed, vectors in model_list["data"]:
        if seed == 1:
            for index, vector in zip(vectors["index"], vectors["vector"]):
                vectors_list.append(vector)
                feature_sets.append(format_key_value_pairs(index, True, prefix="key", dataframe=dataframe_with_keys))
    
    return np.array(vectors_list), feature_sets


def find_top_k_neighbors_for_vector(query_vector, all_vectors, k):
    """
    Find the k nearest neighbors of a query vector (excluding itself).

    Parameters
    ----------
    query_vector : ndarray
        Query vector.
    all_vectors : ndarray
        Array of all vectors.
    k : int
        Number of neighbors to find.

    Returns
    -------
    list
        Indices of the k nearest neighbors.
    """
    similarities = cosine_similarity(query_vector.reshape(1, -1), all_vectors)[0]
    sorted_indices = np.argsort(similarities)[::-1]
    
    neighbors = [idx for idx in sorted_indices if not np.allclose(query_vector, all_vectors[idx])]
    
    return neighbors[:k]


def compute_feature_overlap_fast(index, neighbor_indices, feature_matrix):
    """
    Calculate mean feature overlap between a sample and its neighbors.

    Parameters
    ----------
    index : int
        Index of the target sample.
    neighbor_indices : array-like
        Indices of neighbor samples.
    feature_matrix : ndarray
        Binary feature matrix.

    Returns
    -------
    float
        Mean Jaccard similarity between target and neighbors.
    """
    target_features = feature_matrix[index]
    neighbor_features = feature_matrix[neighbor_indices]
    
    intersection = np.logical_and(neighbor_features, target_features).sum(axis=1)
    union = np.logical_or(neighbor_features, target_features).sum(axis=1)
    
    overlap = np.divide(
        intersection, union,
        out=np.zeros_like(intersection, dtype=float),
        where=union > 0
    )
    return overlap.mean()
