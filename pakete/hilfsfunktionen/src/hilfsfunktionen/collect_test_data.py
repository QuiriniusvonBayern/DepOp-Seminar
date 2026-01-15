from .basic_vector_tests import run_full_quality_check
from .key_vector_tests import run_full_quality_check_key
from .cosine_tests import run_full_vector_test
import pandas as pd

def run_all_tests(model_list, verbose, df_with_keys):
    """
    Führt alle Tests für ein Modell aus und gibt ein kombiniertes Dictionary zurück.
    model_list erwartet [model_object, params_dict]
    """
    
    # 1. Parameter sichern (für spätere Gruppierung in Diagrammen)
    full_result = {
        "params": model_list[1]
    }
    
    # 2. Basic Tests (Semantic + Neighbors)
    # Gibt dict zurück: {"semantic_similarity": [...], "neighbors": [...]}
    full_result["basic"] = run_full_quality_check(model_list, False)
    
    # 3. Key Structure Tests
    if verbose:
        full_result["keys"] = run_full_quality_check_key(model_list, False, df_with_keys=df_with_keys)    
    
    # 4. Vector/Proximity Tests
        full_result["cosine"] = run_full_vector_test(model_list, False, df_with_keys=df_with_keys)
    
    return full_result

# ---------------------------------------------------------
# Hauptschleife zur Datensammlung
# ---------------------------------------------------------

def get_all_results(models, categories, df_with_keys, verbose=False):
    all_model_results = []

    for category in categories:
        print(f"(-------------Test of category: {category.upper()}-------------)\n")
        
        for model_list in models[category]:
            result_data = run_all_tests(
                model_list, 
                verbose=verbose, 
                df_with_keys=df_with_keys
            )
            
            result_data["params"]["Test_Category"] = category
            
            all_model_results.append(result_data)

    print(f"Tests abgeschlossen. {len(all_model_results)} Modelle getestet.")
    return all_model_results
