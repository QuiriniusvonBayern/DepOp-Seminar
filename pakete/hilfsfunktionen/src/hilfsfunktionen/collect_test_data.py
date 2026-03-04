"""
Module for collecting and aggregating test results from semantic SQL model evaluations.
"""

import hashlib
import re
from unittest import result
import pandas as pd
import numpy as np

from .Semantic_SQL_Model_Quality_Score_tests import (
    neighborhood_discriminability_silhouette,
    rank_stability_average,
    feature_coherence_test,
)
from .key_vector_tests import evaluate_model_key_quality


def _extract_suffix(token: str, prefix: str) -> str:
    """
    Extract suffix from tokens like 'Credit_Score_Fair' -> 'Fair'.
    If token does not match prefix, return token unchanged.

    Args:
        token: Input string token
        prefix: Prefix to remove

    Returns:
        Token with prefix removed if present, otherwise original token
    """
    if not isinstance(token, str):
        return str(token)
    if token.startswith(prefix):
        return token[len(prefix) :]
    return token


def derive_fc_labels(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived feature coherence label columns to the dataframe.

    Added columns:
        - fc_risk_tier: Risk category based on credit score
        - fc_value_tier: Value category based on balance and salary
        - fc_engagement_tier: Engagement category based on tenure and activity
        - fc_segment: Combined segment label from all tiers

    Args:
        dataframe: Input dataframe with tokenized features

    Returns:
        Dataframe with added FC label columns
    """
    output_df = dataframe.copy()

    # Risk tier classification from credit score
    credit_scores = output_df["credit_score"].astype(str)

    def _map_risk_category(score_value: str) -> str:
        if "Poor" in score_value or "Fair" in score_value:
            return "FCRISK_High"
        if "Good" in score_value and "Very_Good" not in score_value:
            return "FCRISK_Medium"
        if "Very_Good" in score_value or "Excellent" in score_value:
            return "FCRISK_Low"
        return "FCRISK_Unknown"

    output_df["fc_risk_tier"] = credit_scores.map(_map_risk_category)

    # Value tier classification from balance and salary
    balance_values = output_df["balance"].astype(str)
    salary_values = output_df["estimated_salary"].astype(str)

    def _extract_balance_level(balance_str: str) -> int:
        match = re.search(r"Balance_Cluster_(\d+)", balance_str)
        return int(match.group(1)) if match else -1

    def _extract_salary_level(salary_str: str) -> int:
        level_order = {
            "Very_low": 1,
            "Low": 2,
            "Below_average": 3,
            "Average": 4,
            "Above_average": 5,
            "High": 6,
            "Very_high": 7,
        }
        for key, value in level_order.items():
            if key in salary_str:
                return value
        return -1

    balance_levels = balance_values.map(_extract_balance_level)
    salary_levels = salary_values.map(_extract_salary_level)

    # High value if balance high (>=7) or salary high (>=6)
    # Medium value if balance medium (>=4) or salary medium (>=4)
    # Low value otherwise
    output_df["fc_value_tier"] = pd.Series(
        [
            "FCVAL_High" if (balance >= 7 or salary >= 6) else
            "FCVAL_Medium" if (balance >= 4 or salary >= 4) else
            "FCVAL_Low"
            for balance, salary in zip(balance_levels, salary_levels)
        ],
        index=output_df.index,
    )

    # Engagement tier classification from tenure, activity, and products
    tenure_values = output_df["tenure"].astype(str)
    active_values = output_df["active_member"].astype(str)
    products_values = output_df["products_number"].astype(str)

    def _extract_tenure_int(tenure_str: str) -> int:
        match = re.search(r"Tenure_(\d+)", tenure_str)
        return int(match.group(1)) if match else -1

    def _extract_products_int(products_str: str) -> int:
        match = re.search(r"ProductsNumber_(\d+)", products_str)
        return int(match.group(1)) if match else -1

    tenure_ints = tenure_values.map(_extract_tenure_int)
    products_ints = products_values.map(_extract_products_int)
    is_active = active_values.str.contains("Yes", regex=False)

    # High engagement: active and (tenure >=5 or products >=2)
    # Medium engagement: active or (tenure >=3 or products >=2)
    # Low engagement: otherwise
    output_df["fc_engagement_tier"] = pd.Series(
        [
            "FCENG_High" if (active and (tenure >= 5 or products >= 2)) else
            "FCENG_Medium" if (active or tenure >= 3 or products >= 2) else
            "FCENG_Low"
            for active, tenure, products in zip(is_active, tenure_ints, products_ints)
        ],
        index=output_df.index,
    )

    # Combined segment label for Jaccard-based evaluation
    output_df["fc_segment"] = (
        output_df["fc_risk_tier"] + "|" +
        output_df["fc_value_tier"] + "|" +
        output_df["fc_engagement_tier"]
    ).astype("category")

    return output_df


def build_training_sentences(dataframe: pd.DataFrame, exclude_columns=None) -> list[list[str]]:
    """
    Build sentences for Word2Vec training from a tokenized dataframe.

    Excludes key columns and any explicitly provided columns (e.g., FC label columns).

    Args:
        dataframe: Input dataframe with tokenized features
        exclude_columns: List of column names to exclude

    Returns:
        List of token lists, one per row
    """
    if exclude_columns is None:
        exclude_columns = []

    columns_to_use = []
    for column in dataframe.columns:
        if column in exclude_columns:
            continue
        if column.lower().startswith("key"):
            continue
        columns_to_use.append(column)

    # Convert each row to a list of string tokens
    sentences = dataframe[columns_to_use].astype(str).values.tolist()
    return sentences


def _prepare_model_info(model_info: dict) -> dict:
    """
    Remove seed information from model configuration.

    Args:
        model_info: Original model configuration dictionary

    Returns:
        Configuration dictionary without seed key
    """
    cleaned_info = model_info.copy()
    cleaned_info.pop("Seed", None)
    return cleaned_info


def _run_all_tests(
    model_info: dict,
    sentence_vectors_list: dict,
    verbose: bool,
    key_dataframe: pd.DataFrame,
    model,
    fc_label_dataframe: pd.DataFrame,
) -> dict:
    """
    Execute all evaluation tests for a single model configuration.

    Args:
        model_info: Model configuration parameters
        sentence_vectors_list: Sentence vectors with their configuration
        verbose: Whether to print detailed output
        key_dataframe: Dataframe containing key information
        model: Tuple of (Word2Vec model, configuration with seed)
        fc_label_dataframe: Dataframe with feature coherence labels

    Returns:
        Dictionary containing all test results and metadata
    """
    full_result = {"params": model_info}

    # Raw metrics
    full_result["nd"] = neighborhood_discriminability_silhouette(model, verbose=verbose)

    full_result["rs_mean"], full_result["rs_5"], full_result["rs_10"], full_result["rs_20"] = (
        rank_stability_average(sentence_vectors_list, verbose)
    )

    full_result["fc"] = feature_coherence_test(
        model_list=sentence_vectors_list,
        verbose=verbose,
        dataframe_with_labels=fc_label_dataframe,
    )

    full_result["kvc"] = evaluate_model_key_quality(
        model[0],
        key_dataframe,
        num_keys=50,
        topn=20,
        filter_keys=True,
        max_fetch=10000,
        verbose=verbose,
    )

    return full_result


def _aggregate_vectors_by_model_info(average_sentence_vectors: dict) -> list:
    """
    Aggregate sentence vectors across seeds, grouping by model configuration only.

    Args:
        average_sentence_vectors: Nested dictionary of vectors organized by seed

    Returns:
        List of dictionaries, each containing:
            - model_info: Configuration without seed
            - data: List of (seed, vectors) tuples
    """
    aggregation = {}

    for seed, model_entries in average_sentence_vectors.items():
        for vectors, raw_model_info in model_entries:
            model_info = _prepare_model_info(raw_model_info)
            model_info_key = tuple(sorted(model_info.items()))

            if model_info_key not in aggregation:
                aggregation[model_info_key] = {"model_info": model_info, "data": []}

            aggregation[model_info_key]["data"].append((seed, vectors))

    return list(aggregation.values())


def _aggregate_vectors_by_category_and_model_info(average_sentence_vectors: dict) -> dict:
    """
    Aggregate sentence vectors across seeds, grouping by category and model configuration.

    Args:
        average_sentence_vectors: Nested dictionary of vectors organized by seed and category

    Returns:
        Dictionary mapping categories to lists of entries, each containing:
            - model_info: Configuration without seed
            - data: List of (seed, vectors) tuples
    """
    aggregation = {}

    for seed, categories in average_sentence_vectors.items():
        for category, model_entries in categories.items():
            for vectors, raw_model_info in model_entries:
                model_info = _prepare_model_info(raw_model_info)

                if category not in aggregation:
                    aggregation[category] = {}

                model_info_key = tuple(sorted(model_info.items()))

                if model_info_key not in aggregation[category]:
                    aggregation[category][model_info_key] = {"model_info": model_info, "data": []}

                aggregation[category][model_info_key]["data"].append((seed, vectors))

    # Convert inner dictionaries to lists
    result = {category: list(models.values()) for category, models in aggregation.items()}
    return result


def get_all_grid_results(
    average_sentence_vectors: dict,
    models: dict,
    key_dataframe: pd.DataFrame,
    verbose: bool = False,
) -> list:
    """
    Collect test results for all model configurations (grid search without categories).

    Args:
        average_sentence_vectors: Sentence vectors organized by seed
        models: Dictionary of trained models
        key_dataframe: Dataframe with key information
        verbose: Whether to print detailed output

    Returns:
        List of result dictionaries for all model configurations
    """
    all_results = []
    models_seed_1 = models[1]
    fc_label_dataframe = derive_fc_labels(key_dataframe)
    grouped_vectors = _aggregate_vectors_by_model_info(average_sentence_vectors)

    for vectors_entry, model in zip(grouped_vectors, models_seed_1):
        if len(grouped_vectors) != len(models_seed_1):
            raise ValueError(
                f"Length mismatch: grouped_vectors={len(grouped_vectors)}, "
                f"models_seed_1={len(models_seed_1)}"
            )

        result_data = _run_all_tests(
            model_info=vectors_entry["model_info"],
            sentence_vectors_list=vectors_entry,
            model=model,
            verbose=verbose,
            key_dataframe=key_dataframe,
            fc_label_dataframe=fc_label_dataframe,
        )

        # Store complete sentence vectors with their configuration
        result_data["used_sentence_vectors"] = {
            "model_info": vectors_entry["model_info"],
            "data": vectors_entry["data"],
        }

        # Store complete model with seed information
        result_data["used_model"] = {
            "model_object": model[0],
            "model_config_with_seed": model[1],
        }

        # Generate unique experiment identifier
        config_string = str(model[1]) + str(vectors_entry["model_info"])
        result_data["experiment_id"] = hashlib.md5(config_string.encode()).hexdigest()[:10]

        all_results.append(result_data)

    # Normalization and SSMQ calculation
    nd_values = np.array([result["nd"] for result in all_results])
    fc_values = np.array([result["fc"] for result in all_results])
    rs_values = np.array([result["rs_mean"] for result in all_results])

    nd_min = nd_values.min()
    nd_max = nd_values.max()

    for result in all_results:
        result["nd_norm"] = (result["nd"] - nd_min) / (nd_max - nd_min)

    for result in all_results:
        result["nd"] = result["nd_norm"]

    alpha, beta, gamma = 0.3, 0.5, 0.2

    for result in all_results:
        result["ssmq"] = (
            alpha * result["nd_norm"] + beta * result["rs_mean"] + gamma * result["fc"]
        )

    # Add summary information
    for result in all_results:
        result["summary"] = {
            "vector_size": result["params"]["Vector Size"],
            "algorithm": result["params"]["Algorithmus"],
            "window": result["params"]["Window"],
            "epochs": result["params"]["Epochs"],
            "seed": result["used_model"]["model_config_with_seed"].get("Seed", "N/A"),
            "nd_score": float(result["nd"]),
            "rs_mean_score": float(result["rs_mean"]),
            "fc_score": float(result["fc"]),
            "ssmq_score": float(result["ssmq"]),
            "kvc_recall": float(result["kvc"]["overall_recall"]),
        }

    print(f"Testing complete. {len(all_results)} models evaluated.")

    if verbose:
        print("\nStored data structure per result:")
        print("1. params: Configuration (without Test_Category)")
        print("2. used_sentence_vectors: Vector data + configuration (without seed)")
        print("3. used_model: Model object + configuration (with seed)")
        print("4. Metrics: nd, rs_mean, fc, kvc, ssmq")
        print("5. experiment_id: Unique identifier")
        print("6. summary: Key metrics for overview (without test_category)")

    return all_results


def get_all_sweep_results(
    average_sentence_vectors: dict,
    models: dict,
    key_dataframe: pd.DataFrame,
    verbose: bool = False,
) -> list:
    """
    Collect test results for all model configurations (sweep with categories).

    Args:
        average_sentence_vectors: Sentence vectors organized by seed and category
        models: Dictionary of trained models
        key_dataframe: Dataframe with key information
        verbose: Whether to print detailed output

    Returns:
        List of result dictionaries for all model configurations
    """
    all_results = []
    models_seed_1 = models[1]
    grouped_vectors = _aggregate_vectors_by_category_and_model_info(average_sentence_vectors)
    fc_label_dataframe = derive_fc_labels(key_dataframe)

    for category, vectors_entries in grouped_vectors.items():
        print(f"(------------- Testing category: {category.upper()} -------------)\n")

        for vectors_entry, model in zip(vectors_entries, models_seed_1[category]):
            if len(vectors_entries) != len(models_seed_1[category]):
                raise ValueError(
                    f"Length mismatch: vectors_entries={len(vectors_entries)}, "
                    f"models_seed_1[{category}]={len(models_seed_1[category])}"
                )

            result_data = _run_all_tests(
                model_info=vectors_entry["model_info"],
                sentence_vectors_list=vectors_entry,
                model=model,
                verbose=verbose,
                key_dataframe=key_dataframe,
                fc_label_dataframe=fc_label_dataframe,
            )

            # Add test category to parameters
            result_data["params"]["Test_Category"] = category

            # Store complete sentence vectors with their configuration
            result_data["used_sentence_vectors"] = {
                "model_info": vectors_entry["model_info"],
                "data": vectors_entry["data"],
            }

            # Store complete model with seed information
            result_data["used_model"] = {
                "model_object": model[0],
                "model_config_with_seed": model[1],
            }

            # Generate unique experiment identifier
            config_string = str(model[1]) + str(vectors_entry["model_info"])
            result_data["experiment_id"] = hashlib.md5(config_string.encode()).hexdigest()[:10]

            all_results.append(result_data)

    # Normalization and SSMQ calculation
    nd_values = np.array([result["nd"] for result in all_results])
    fc_values = np.array([result["fc"] for result in all_results])
    rs_values = np.array([result["rs_mean"] for result in all_results])

    nd_min = nd_values.min()
    nd_max = nd_values.max()

    for result in all_results:
        result["nd_norm"] = (result["nd"] - nd_min) / (nd_max - nd_min)

    for result in all_results:
        result["nd"] = result["nd_norm"]

    alpha, beta, gamma = 0.3, 0.5, 0.2

    for result in all_results:
        result["ssmq"] = (
            alpha * result["nd_norm"] + beta * result["rs_mean"] + gamma * result["fc"]
        )

    # Add summary information
    for result in all_results:
        result["summary"] = {
            "test_category": result["params"]["Test_Category"],
            "vector_size": result["params"]["Vector Size"],
            "algorithm": result["params"]["Algorithmus"],
            "window": result["params"]["Window"],
            "epochs": result["params"]["Epochs"],
            "seed": result["used_model"]["model_config_with_seed"].get("Seed", "N/A"),
            "nd_score": float(result["nd"]),
            "rs_mean_score": float(result["rs_mean"]),
            "fc_score": float(result["fc"]),
            "ssmq_score": float(result["ssmq"]),
            "kvc_recall": float(result["kvc"]["overall_recall"]),
        }

    print(f"Testing complete. {len(all_results)} models evaluated.")

    if verbose:
        print("\nStored data structure per result:")
        print("1. params: Configuration + Test_Category")
        print("2. used_sentence_vectors: Vector data + configuration (without seed)")
        print("3. used_model: Model object + configuration (with seed)")
        print("4. Metrics: nd, rs_mean, fc, kvc, ssmq")
        print("5. experiment_id: Unique identifier")
        print("6. summary: Key metrics for overview")

    return all_results
