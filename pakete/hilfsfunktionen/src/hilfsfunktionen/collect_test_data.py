from unittest import result
from .Semantic_SQL_Model_Quality_Score_tests import neighborhood_discriminability_test, rank_stability_test_average, feature_coherence_test
from .key_vector_tests import evaluate_model_key_quality
import pandas as pd
import numpy as np

def run_all_tests(info, average_sentence_vectors_list, verbose, df_with_keys, model):
    full_result = {
        "params": info
    }

    # Rohmetriken
    full_result["nd"] = neighborhood_discriminability_test(model, verbose)
    full_result["rs_mean"], full_result["rs_5"], full_result["rs_10"], full_result["rs_20"] = \
        rank_stability_test_average(average_sentence_vectors_list, verbose)
    full_result["fc"] = feature_coherence_test(
        average_sentence_vectors_list,
        verbose,
        df_with_keys=df_with_keys
    )

    full_result["kvc"] = evaluate_model_key_quality(
        model[0], df_with_keys, 
        num_keys=50, topn=20, 
        filter_keys=True, 
        max_fetch=10000, 
        verbose=verbose
    )

    return full_result


# ---------------------------------------------------------
# Hauptschleife zur Datensammlung
# ---------------------------------------------------------

def get_all_sweep_results(average_sentence_vectors, models, df_with_keys, verbose=False):
    all_model_results = []
    models_seed_1 = models[1]

    grouped_vectors = aggregate_vectors_by_category_and_model_info(average_sentence_vectors)

    for category, model_info in grouped_vectors.items():
        print(f"(-------------Test of category: {category.upper()}-------------)\n")
        for average_sentence_vectors_list, model in zip(model_info, models_seed_1[category]):
            if len(model_info) != len(models_seed_1[category]):
                raise ValueError(f"Listen haben unterschiedliche Längen: "
                                f"model_info={len(model_info)}, "
                                f"models_seed_1[category]={len(models_seed_1[category])}")

            
            result_data = run_all_tests(
                info = average_sentence_vectors_list["model_info"],
                average_sentence_vectors_list = average_sentence_vectors_list, 
                model = model,  
                verbose=verbose, 
                df_with_keys=df_with_keys
            )
            
            result_data["params"]["Test_Category"] = category
            
            all_model_results.append(result_data)

    nd_values = np.array([r["nd"] for r in all_model_results])
    fc_values = np.array([r["fc"] for r in all_model_results])
    rs_values = np.array([r["rs_mean"] for r in all_model_results])

    nd_min = nd_values.min()
    nd_max = nd_values.max()

    for r in all_model_results:
        r["nd_norm"] = (r["nd"] - nd_min) / (nd_max - nd_min)

    for r in all_model_results:
        r["nd"] = r["nd_norm"]

    alpha, beta, gamma = 0.5, 0.3, 0.2

    for r in all_model_results:
        r["ssmq"] = (
            alpha * r["nd_norm"] +
            beta  * r["rs_mean"] +
            gamma * r["fc"]
        )


    print(f"Tests abgeschlossen. {len(all_model_results)} Modelle getestet.")
    return all_model_results


def aggregate_vectors_by_category_and_model_info(average_sentence_vectors):
    """
    Aggregiert average_sentence_vectors über alle Seeds hinweg.
    Gruppierung erfolgt ausschließlich über model_info.
    
    Rückgabe:
        dict:
            key   -> category
            value -> Liste von Dicts:
                     {
                         "model_info": model_info,
                         "data": [(seed, vectors), ...]
                     }
    """
    
    aggregation = {}
    
    for seed, categories in average_sentence_vectors.items():
        for category, model_entries in categories.items():
            for vectors, model_info_uncleaned in model_entries:
                model_info = prepare_model_info(model_info_uncleaned)
                if category not in aggregation:
                    aggregation[category] = {}

                model_info_key = tuple(sorted(model_info.items()))

                if model_info_key not in aggregation[category]:
                    aggregation[category][model_info_key] = {
                        "model_info": model_info,
                        "data": []
                    }

                aggregation[category][model_info_key]["data"].append(
                    (seed, vectors)
                )
    
    result = {
        category: list(models.values())
        for category, models in aggregation.items()
    }
    
    return result

def prepare_model_info(model_info):
    cleaned_info = model_info.copy()
    cleaned_info.pop('Seed', None)  
    return cleaned_info



def get_all_grid_results(average_sentence_vectors, models, df_with_keys, verbose=False):
    all_model_results = []

    models_seed_1 = models[1]  # [(model_object, model_info), ...]

    grouped_vectors = aggregate_vectors_by_model_info(average_sentence_vectors)

    if len(grouped_vectors) != len(models_seed_1):
        raise ValueError(
            f"Listen haben unterschiedliche Längen: "
            f"grouped_vectors={len(grouped_vectors)}, "
            f"models_seed_1={len(models_seed_1)}"
        )

    for vector_entry, model_tuple in zip(grouped_vectors, models_seed_1):
        # model_tuple == (Word2Vec, model_info)
        result_data = run_all_tests(
            info=vector_entry["model_info"],
            average_sentence_vectors_list=vector_entry,
            model=model_tuple,   # <<< WICHTIG: komplettes Tupel weiterreichen
            verbose=verbose,
            df_with_keys=df_with_keys
        )


        all_model_results.append(result_data)

        nd_values = np.array([r["nd"] for r in all_model_results])
        fc_values = np.array([r["fc"] for r in all_model_results])
        rs_values = np.array([r["rs_mean"] for r in all_model_results])

        nd_min = nd_values.min()
        nd_max = nd_values.max()

        for r in all_model_results:
            r["nd_norm"] = (r["nd"] - nd_min) / (nd_max - nd_min)

        alpha, beta, gamma = 0.5, 0.3, 0.2

        for r in all_model_results:
            r["ssmq"] = (
                alpha * r["nd_norm"] +
                beta  * r["rs_mean"] +
                gamma * r["fc"]
            )

    print(f"Tests abgeschlossen. {len(all_model_results)} Modelle getestet.")
    return all_model_results




def aggregate_vectors_by_model_info(average_sentence_vectors):
    """
    Aggregiert average_sentence_vectors über alle Seeds hinweg.
    Gruppierung erfolgt ausschließlich über model_info (ohne Kategorien).

    Rückgabe:
        Liste von Dicts:
            {
                "model_info": model_info,
                "data": [(seed, vectors), ...]
            }
    """
    aggregation = {}

    for seed, model_entries in average_sentence_vectors.items():
        for vectors, model_info_uncleaned in model_entries:
            model_info = prepare_model_info(model_info_uncleaned)
            model_info_key = tuple(sorted(model_info.items()))

            if model_info_key not in aggregation:
                aggregation[model_info_key] = {
                    "model_info": model_info,
                    "data": []
                }

            aggregation[model_info_key]["data"].append(
                (seed, vectors)
            )
    return list(aggregation.values())
