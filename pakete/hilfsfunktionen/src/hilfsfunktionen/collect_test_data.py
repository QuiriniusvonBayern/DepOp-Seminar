from unittest import result
from .Semantic_SQL_Model_Quality_Score_tests import neighborhood_discriminability_test_silhouette, rank_stability_test_average, feature_coherence_test
from .key_vector_tests import evaluate_model_key_quality
import pandas as pd
import numpy as np

def run_all_tests(info, average_sentence_vectors_list, verbose, df_with_keys, model):
    full_result = {
        "params": info
    }

    # Rohmetriken
    full_result["nd"] = neighborhood_discriminability_test_silhouette(
        model,
        verbose=verbose
    )

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
            
            # Füge Test_Category zu params hinzu
            result_data["params"]["Test_Category"] = category
            
            # 1. Behalte die vollständigen Sentence Vectors
            result_data["used_sentence_vectors"] = {
                "model_info": average_sentence_vectors_list["model_info"],  # Konfiguration OHNE Seed
                "data": average_sentence_vectors_list["data"]  # Die kompletten Vektordaten
            }
            
            # 2. Behalte das vollständige Modell mit Seed-Information
            # model ist ein Tuple: (Word2Vec-Objekt, Konfiguration-mit-Seed)
            result_data["used_model"] = {
                "model_object": model[0],  # Das eigentliche Word2Vec-Modell
                "model_config_with_seed": model[1]  # Vollständige Konfiguration MIT Seed
            }
            
            # 3. Entferne redundante model_info (da es bereits in used_sentence_vectors enthalten ist)
            # result_data hat bereits "params" und "used_sentence_vectors" mit allen Informationen
            
            # 4. Optional: Füge eine eindeutige ID für dieses Experiment hinzu
            import hashlib
            config_str = str(model[1]) + str(average_sentence_vectors_list["model_info"])
            result_data["experiment_id"] = hashlib.md5(config_str.encode()).hexdigest()[:10]
            
            all_model_results.append(result_data)

    # Normierung und SSMQ-Berechnung
    nd_values = np.array([r["nd"] for r in all_model_results])
    fc_values = np.array([r["fc"] for r in all_model_results])
    rs_values = np.array([r["rs_mean"] for r in all_model_results])

    nd_min = nd_values.min()
    nd_max = nd_values.max()

    for r in all_model_results:
        r["nd_norm"] = (r["nd"] - nd_min) / (nd_max - nd_min)

    for r in all_model_results:
        r["nd"] = r["nd_norm"]

    alpha, beta, gamma = 0.3, 0.5, 0.2

    for r in all_model_results:
        r["ssmq"] = (
            alpha * r["nd_norm"] +
            beta  * r["rs_mean"] +
            gamma * r["fc"]
        )
    
    # Füge Zusammenfassungsinformationen hinzu
    for r in all_model_results:
        r["summary"] = {
            "test_category": r["params"]["Test_Category"],
            "vector_size": r["params"]["Vector Size"],
            "algorithm": r["params"]["Algorithmus"],
            "window": r["params"]["Window"],
            "epochs": r["params"]["Epochs"],
            "seed": r["used_model"]["model_config_with_seed"].get("Seed", "N/A"),
            "nd_score": float(r["nd"]),
            "rs_mean_score": float(r["rs_mean"]),
            "fc_score": float(r["fc"]),
            "ssmq_score": float(r["ssmq"]),
            "kvc_recall": float(r["kvc"]["overall_recall"])
        }

    print(f"Tests abgeschlossen. {len(all_model_results)} Modelle getestet.")
    
    # Ausgabe der gespeicherten Datenstruktur
    if verbose:
        print(f"\nGespeicherte Datenstruktur pro Ergebnis:")
        print(f"1. params: Konfiguration + Test_Category")
        print(f"2. used_sentence_vectors: Vektordaten + Konfiguration (ohne Seed)")
        print(f"3. used_model: Modellobjekt + Konfiguration (mit Seed)")
        print(f"4. Metriken: nd, rs_mean, fc, kvc, ssmq")
        print(f"5. Experiment-ID: Eindeutige Kennung")
        print(f"6. summary: Wichtige Metriken für Übersichten")
    
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
    models_seed_1 = models[1]

    grouped_vectors = aggregate_vectors_by_model_info(average_sentence_vectors)

    for average_sentence_vectors_list, model in zip(grouped_vectors, models_seed_1):
        if len(grouped_vectors) != len(models_seed_1):
            raise ValueError(f"Listen haben unterschiedliche Längen: "
                            f"grouped_vectors={len(grouped_vectors)}, "
                            f"models_seed_1={len(models_seed_1)}")

        result_data = run_all_tests(
            info = average_sentence_vectors_list["model_info"],
            average_sentence_vectors_list = average_sentence_vectors_list, 
            model = model,  
            verbose=verbose, 
            df_with_keys=df_with_keys
        )
        
        # KEINE Test_Category - das ist der einzige Unterschied
        # result_data["params"]["Test_Category"] = category  # <-- Diese Zeile fehlt bewusst
        
        # 1. Behalte die vollständigen Sentence Vectors
        result_data["used_sentence_vectors"] = {
            "model_info": average_sentence_vectors_list["model_info"],  # Konfiguration OHNE Seed
            "data": average_sentence_vectors_list["data"]  # Die kompletten Vektordaten
        }
        
        # 2. Behalte das vollständige Modell mit Seed-Information
        # model ist ein Tuple: (Word2Vec-Objekt, Konfiguration-mit-Seed)
        result_data["used_model"] = {
            "model_object": model[0],  # Das eigentliche Word2Vec-Modell
            "model_config_with_seed": model[1]  # Vollständige Konfiguration MIT Seed
        }
        
        # 3. Entferne redundante model_info (da es bereits in used_sentence_vectors enthalten ist)
        # result_data hat bereits "params" und "used_sentence_vectors" mit allen Informationen
        
        # 4. Optional: Füge eine eindeutige ID für dieses Experiment hinzu
        import hashlib
        config_str = str(model[1]) + str(average_sentence_vectors_list["model_info"])
        result_data["experiment_id"] = hashlib.md5(config_str.encode()).hexdigest()[:10]
        
        all_model_results.append(result_data)

    # Normierung und SSMQ-Berechnung
    nd_values = np.array([r["nd"] for r in all_model_results])
    fc_values = np.array([r["fc"] for r in all_model_results])
    rs_values = np.array([r["rs_mean"] for r in all_model_results])

    nd_min = nd_values.min()
    nd_max = nd_values.max()

    for r in all_model_results:
        r["nd_norm"] = (r["nd"] - nd_min) / (nd_max - nd_min)

    for r in all_model_results:
        r["nd"] = r["nd_norm"]

    alpha, beta, gamma = 0.3, 0.5, 0.2

    for r in all_model_results:
        r["ssmq"] = (
            alpha * r["nd_norm"] +
            beta  * r["rs_mean"] +
            gamma * r["fc"]
        )
    
    # Füge Zusammenfassungsinformationen hinzu
    for r in all_model_results:
        r["summary"] = {
            # "test_category": r["params"]["Test_Category"],  # <-- Diese Zeile fehlt bewusst
            "vector_size": r["params"]["Vector Size"],
            "algorithm": r["params"]["Algorithmus"],
            "window": r["params"]["Window"],
            "epochs": r["params"]["Epochs"],
            "seed": r["used_model"]["model_config_with_seed"].get("Seed", "N/A"),
            "nd_score": float(r["nd"]),
            "rs_mean_score": float(r["rs_mean"]),
            "fc_score": float(r["fc"]),
            "ssmq_score": float(r["ssmq"]),
            "kvc_recall": float(r["kvc"]["overall_recall"])
        }

    print(f"Tests abgeschlossen. {len(all_model_results)} Modelle getestet.")
    
    # Ausgabe der gespeicherten Datenstruktur
    if verbose:
        print(f"\nGespeicherte Datenstruktur pro Ergebnis:")
        print(f"1. params: Konfiguration (OHNE Test_Category)")
        print(f"2. used_sentence_vectors: Vektordaten + Konfiguration (ohne Seed)")
        print(f"3. used_model: Modellobjekt + Konfiguration (mit Seed)")
        print(f"4. Metriken: nd, rs_mean, fc, kvc, ssmq")
        print(f"5. Experiment-ID: Eindeutige Kennung")
        print(f"6. summary: Wichtige Metriken für Übersichten (OHNE test_category)")
    
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
