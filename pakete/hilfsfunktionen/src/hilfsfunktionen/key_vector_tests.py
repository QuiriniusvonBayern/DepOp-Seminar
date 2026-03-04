import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set


def get_key_values(key_index: int, filter_strings: bool, exclude_prefix: str, df_with_keys: pd.DataFrame, include_churn: bool = False) -> List[str]:
    """Extract values from a specific row based on the given key index."""
    row_values = df_with_keys.iloc[key_index]
    
    if not include_churn:
        row_values = [str(val) for val in row_values if isinstance(val, str) and not str(val).startswith("Churn_")]
    
    if filter_strings:
        return [str(val) for val in row_values if isinstance(val, str) and not str(val).startswith(exclude_prefix)]
    else:
        return [str(val) for val in row_values]


def compare_lists(list_a: List, list_b: List) -> Tuple[str, int]:
    """Compare two lists element-wise and return matching elements with positions."""
    output_message = ""
    matching_pairs = [(idx, val_a, val_b) for idx, (val_a, val_b) in enumerate(zip(list_a, list_b)) if val_a == val_b]
    match_count = len(matching_pairs)
    
    output_message += f"{match_count} of {len(list_a)} matching values: "
    for position, val_a, val_b in matching_pairs:
        output_message += f"(Position {position} : '{val_a}')"
    
    return output_message, match_count


def has_churned(customer_index: int, df: pd.DataFrame) -> bool:
    """
    Determine whether a customer has churned based on their row index.

    Args:
        customer_index: Zero-based index of the customer row.
        df: DataFrame containing customer data with a 'churn' column.

    Returns:
        True if the customer has churned (churn == 'Churn_Yes'), False otherwise.

    Raises:
        IndexError: If the customer index is out of bounds or the 'churn' column is missing.
    """
    if customer_index < 0 or customer_index >= len(df):
        raise IndexError(f"Customer index {customer_index} out of bounds for [0, {len(df)-1}]")
    if 'churn' not in df.columns:
        raise IndexError("Column 'churn' not in the DataFrame.")

    churn_value = df.iloc[customer_index]['churn']

    if pd.isna(churn_value):
        return False
    
    return "Churn_Yes" == churn_value


def is_female(customer_index: int, df: pd.DataFrame) -> bool:
    """
    Determine whether a customer is female based on their row index.

    Args:
        customer_index: Zero-based index of the customer row.
        df: DataFrame containing customer data with a 'gender' column.

    Returns:
        True if the customer is female ('Gender_Female'), False otherwise.

    Raises:
        IndexError: If the customer index is out of bounds or the 'gender' column is missing.
    """
    if customer_index < 0 or customer_index >= len(df):
        raise IndexError(f"Customer index {customer_index} out of bounds for [0, {len(df)-1}]")
    if 'gender' not in df.columns:
        raise IndexError("Column 'gender' not in the DataFrame.")

    gender_value = df.iloc[customer_index]['gender']

    if pd.isna(gender_value):
        return False
    
    return "Gender_Female" == gender_value


def get_country_code(customer_index: int, df: pd.DataFrame) -> int:
    """
    Get the country code for a customer based on their row index.

    Args:
        customer_index: Zero-based index of the customer row.
        df: DataFrame containing customer data with a 'country' column.

    Returns:
        0 for France, 1 for Germany, 2 for Spain, 5 for unknown or missing.

    Raises:
        IndexError: If the customer index is out of bounds or the 'country' column is missing.
    """
    if customer_index < 0 or customer_index >= len(df):
        raise IndexError(f"Customer index {customer_index} out of bounds for [0, {len(df)-1}]")
    if 'country' not in df.columns:
        raise IndexError("Column 'country' not in the DataFrame.")

    country_value = df.iloc[customer_index]['country']

    if pd.isna(country_value):
        return 5
    
    match country_value:
        case "Country_France":
            return 0
        case "Country_Germany":
            return 1
        case "Country_Spain":
            return 2

    return 5


def get_row_values(key_index: int, filter_keys: bool = True, exclude_prefix: str = "key", df_with_keys: pd.DataFrame = None) -> List[str]:
    """Extract values from a specific row based on the key index, optionally filtering out keys."""
    row_values = df_with_keys.iloc[key_index]
    
    if filter_keys:
        return [val for val in row_values if not val.startswith(exclude_prefix)]
    else:
        return [val for val in row_values]


def calculate_key_to_values_score(model, key: str, df_with_keys: pd.DataFrame, topn: int = 20,
                                   filter_keys: bool = True, max_fetch: int = 1000,
                                   verbose: bool = False) -> Dict:
    """
    Measure how well a key encodes the semantic information of its corresponding row.

    A good key should be close in the vector space to its own row values
    (e.g., key_1 close to Age_Young_Seniors, Gender_Male, etc.).

    Args:
        model: Word2Vec model.
        key: Key string (e.g., "key_1").
        df_with_keys: DataFrame containing keys and values.
        topn: Number of desired non-key neighbors.
        filter_keys: If True, filter other keys from neighbors.
        max_fetch: Maximum initial neighbors to fetch when filtering.
        verbose: If True, print detailed analysis.

    Returns:
        Dictionary containing proximity scores and detailed information.
    """
    key_index = int(key.replace("key_", ""))
    row_values = get_row_values(key_index, filter_keys=True, df_with_keys=df_with_keys)
    
    row_values_in_vocab = [val for val in row_values if val in model.wv]
    
    if len(row_values_in_vocab) == 0:
        return {
            'error': f'No row values from {key} found in model vocabulary',
            'key': key,
            'total_values': len(row_values)
        }
    
    try:
        if filter_keys:
            all_neighbors = model.wv.most_similar(key, topn=max_fetch)
            
            non_key_neighbors = [(word, sim) for word, sim in all_neighbors 
                                  if not word.startswith("key_")]
            
            neighbors = non_key_neighbors[:topn]
            
            if verbose:
                total_fetched = len(all_neighbors)
                keys_filtered = total_fetched - len(non_key_neighbors)
                actual_retrieved = len(neighbors)
                
                print(f"ℹ️  Neighbor statistics:")
                print(f"   Initially fetched:        {total_fetched}")
                print(f"   Keys filtered out:        {keys_filtered} ({keys_filtered/total_fetched*100:.1f}%)")
                print(f"   Remaining non-keys:       {len(non_key_neighbors)}")
                print(f"   Top-N used:               {actual_retrieved}")
                
                if actual_retrieved < topn:
                    print(f"   ⚠️  Warning: Only {actual_retrieved}/{topn} non-key neighbors found!")
                    print(f"      Increase max_fetch (currently {max_fetch}) for better results.")
        else:
            neighbors = model.wv.most_similar(key, topn=topn)
            
    except KeyError:
        return {'error': f'{key} not in vocabulary'}
    
    neighbor_words = {word for word, _ in neighbors}
    
    found_values = []
    missing_values = []
    
    for value in row_values_in_vocab:
        if value in neighbor_words:
            rank = next(i for i, (word, _) in enumerate(neighbors) if word == value)
            similarity = neighbors[rank][1]
            found_values.append({
                'value': value,
                'rank': rank,
                'similarity': similarity,
                'weight': 1.0 / (rank + 1)
            })
        else:
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
    
    proximity_score = len(found_values) / len(row_values_in_vocab)
    
    if len(found_values) > 0:
        weights = [item['weight'] for item in found_values]
        weight_sum = sum(weights)
        weighted_proximity_score = weight_sum / len(row_values_in_vocab)
    else:
        weighted_proximity_score = 0.0
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Analysis for {key} (row {key_index})")
        print(f"{'='*60}")
        print(f"Filter keys: {'Yes' if filter_keys else 'No'}")
        if filter_keys:
            print(f"Max fetch: {max_fetch}")
        print(f"Row values in vocabulary: {len(row_values_in_vocab)}/{len(row_values)}")
        print(f"\nValues found in top-{len(neighbors)} neighbors: {len(found_values)}/{len(row_values_in_vocab)}")
        
        if found_values:
            print("\n✓ Found row values:")
            for item in sorted(found_values, key=lambda x: x['rank']):
                print(f"  Rank {item['rank']:2d}: {item['value']:30s} (Sim: {item['similarity']:.4f})")
        
        if missing_values and verbose:
            print(f"\n✗ Missing row values (not in top-{topn}):")
            for item in sorted(missing_values, key=lambda x: x['similarity'] if x['similarity'] else -1, reverse=True)[:5]:
                sim_str = f"{item['similarity']:.4f}" if item['similarity'] else "N/A"
                print(f"  {item['value']:30s} (Sim: {sim_str})")
        
        print(f"\nProximity score:          {proximity_score:.3f}")
        print(f"Weighted proximity score: {weighted_proximity_score:.3f}")
    
    return {
        'key': key,
        'proximity_score': proximity_score,
        'weighted_proximity_score': weighted_proximity_score,
        'found_values': len(found_values),
        'total_values_in_vocab': len(row_values_in_vocab),
        'total_values': len(row_values),
        'missing_values': len(missing_values),
        'found_details': found_values,
        'missing_details': missing_values,
        'topn': topn,
        'filter_keys': filter_keys,
        'max_fetch': max_fetch if filter_keys else None,
        'actual_neighbors_count': len(neighbors)
    }


def evaluate_model_key_quality(model, df_with_keys: pd.DataFrame, num_keys: int = 10, topn: int = 20,
                                filter_keys: bool = True, max_fetch: int = 1000,
                                verbose: bool = False) -> Dict:
    """
    Evaluate key quality across multiple keys.

    Measures how well keys encode the semantic information of their rows by checking
    if row values are close to the key in the vector space.

    Args:
        model: Word2Vec model.
        df_with_keys: DataFrame containing keys and values.
        num_keys: Number of keys to sample for evaluation.
        topn: Number of neighbors to consider.
        filter_keys: If True, filter other keys from neighbors.
        max_fetch: Maximum initial neighbors to fetch when filtering.
        verbose: If True, print detailed evaluation.

    Returns:
        Dictionary with aggregated metrics and interpretations.
    """
    available_keys = [f"key_{i}" for i in range(len(df_with_keys))]
    
    valid_keys = [key for key in available_keys if key in model.wv]
    
    if len(valid_keys) == 0:
        return {'error': 'No valid keys found in model'}
    
    sample_size = min(num_keys, len(valid_keys))
    sampled_keys = np.random.choice(valid_keys, size=sample_size, replace=False)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"  WORD2VEC KEY-TO-VALUES PROXIMITY EVALUATION")
        print(f"{'='*70}")
        print(f"Evaluating {sample_size} keys with top-{topn} non-key neighbors")
        print(f"Filter keys from neighbors: {'Yes ✓' if filter_keys else 'No ✗'}")
        if filter_keys:
            print(f"Max fetch per key: {max_fetch}")
        print(f"Hypothesis: Keys should be close to their own row values")
        print(f"{'='*70}")
    
    results = []
    for key in sampled_keys:
        result = calculate_key_to_values_score(model, key, df_with_keys, topn, filter_keys, max_fetch, verbose)
        
        if 'error' not in result:
            results.append(result)
    
    if len(results) == 0:
        return {'error': 'No evaluable keys found'}
    
    avg_proximity = np.mean([res['proximity_score'] for res in results])
    avg_weighted_proximity = np.mean([res['weighted_proximity_score'] for res in results])
    total_found = sum([res['found_values'] for res in results])
    total_possible = sum([res['total_values_in_vocab'] for res in results])
    
    summary = {
        'model_score': avg_weighted_proximity,
        'avg_proximity_score': avg_proximity,
        'avg_weighted_proximity_score': avg_weighted_proximity,
        'total_found_values': total_found,
        'total_possible_values': total_possible,
        'overall_recall': total_found / total_possible if total_possible > 0 else 0,
        'keys_evaluated': len(results),
        'topn': topn,
        'filter_keys': filter_keys,
        'max_fetch': max_fetch if filter_keys else None,
        'interpretation': _interpret_proximity_score(avg_weighted_proximity),
    }
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"  SUMMARY")
        print(f"{'='*70}")
        print(f"Filter keys enabled:       {'Yes ✓' if filter_keys else 'No ✗'}")
        if filter_keys:
            print(f"Max fetch used:            {max_fetch}")
        print(f"Model score (weighted):     {avg_weighted_proximity:.3f}")
        print(f"Average proximity score:    {avg_proximity:.3f}")
        print(f"Overall recall:             {summary['overall_recall']:.3f} ({total_found}/{total_possible})")
        print(f"Keys evaluated:             {len(results)}")
        print(f"Interpretation:             {summary['interpretation']}")
        print(f"{'='*70}\n")
    
    return summary


def _interpret_proximity_score(score: float) -> str:
    """Provide a textual interpretation of a proximity score."""
    if score >= 0.7:
        return "Excellent - Keys encode their row values very well"
    elif score >= 0.5:
        return "Good - Keys show strong proximity to their row values"
    elif score >= 0.3:
        return "Acceptable - Moderate proximity to row values"
    elif score >= 0.15:
        return "Weak - Low proximity to own row values"
    else:
        return "Insufficient - Keys barely encode their row values"


def analyze_key_specificity(model, key: str, df_with_keys: pd.DataFrame, topn: int = 50) -> Dict:
    """
    Analyze how specifically a key represents its row.

    Measures the ratio of:
    - Own row values among neighbors (good)
    - Foreign row values among neighbors (bad - too generic)
    - Other tokens among neighbors (neutral)
    """
    key_index = int(key.replace("key_", ""))
    own_values = set(get_row_values(key_index, filter_keys=True, df_with_keys=df_with_keys))
    
    all_row_values = set()
    for idx in range(len(df_with_keys)):
        row_values = get_row_values(idx, filter_keys=True, df_with_keys=df_with_keys)
        all_row_values.update(row_values)
    
    foreign_values = all_row_values - own_values
    
    try:
        neighbors = model.wv.most_similar(key, topn=topn)
    except KeyError:
        return {'error': f'{key} not in vocabulary'}
    
    own_count = 0
    foreign_count = 0
    other_count = 0
    
    for word, _ in neighbors:
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
        'interpretation': 'Specific' if specificity_score > 0.2 else 'Generic' if specificity_score < -0.2 else 'Neutral'
    }


def compare_models(models_dict: Dict, df_with_keys: pd.DataFrame, num_keys: int = 10, topn: int = 20,
                   filter_keys: bool = True, max_fetch: int = 1000) -> pd.DataFrame:
    """
    Compare multiple Word2Vec models based on their key quality.

    Args:
        models_dict: Dictionary with model names as keys and Word2Vec models as values.
        df_with_keys: DataFrame containing keys and values.
        num_keys: Number of keys to sample for evaluation.
        topn: Number of neighbors to consider.
        filter_keys: If True, filter other keys from neighbors.
        max_fetch: Maximum initial neighbors to fetch when filtering.

    Returns:
        DataFrame with comparison results, sorted by quality score.
    """
    comparison_results = []
    
    for model_name, model in models_dict.items():
        print(f"\nEvaluating model: {model_name}")
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
