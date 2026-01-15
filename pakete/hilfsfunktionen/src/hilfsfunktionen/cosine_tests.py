from .cosine_funktions import proximity_avg, proximity_topn_avg, subset_proximity_avg, proximity_avg_all
from .key_vector_tests import get_key_values, compare_lists
from datetime import timedelta
import time

# ----- subset_proximity_avg proximity_avg_all proximity_topn_avg proximity_avg ---
def find_n_most_similar_to_key(model, key, method, n=5, df_with_keys=None):
    vectors_key = []
    vectors_temp = []
    most_similar = []
    vectors_key = get_vector_list(model, key, df_with_keys=df_with_keys)
    for i in range(len(df_with_keys)):
        if i == key:
            continue
        vectors_temp = get_vector_list(model, i, df_with_keys=df_with_keys)
        match method:
            case "proximity_avg":
                avg_value = proximity_avg(vectors_key, vectors_temp)
            case "proximity_topn_avg":
                avg_value = proximity_topn_avg(vectors_key, vectors_temp)
            case "subset_proximity_avg":
                avg_value = subset_proximity_avg(vectors_key, vectors_temp)
            case "proximity_avg_all":
                avg_value = proximity_avg_all(vectors_key, vectors_temp)
            case _:
                return 'Unbekanntes Kommando'
        
        most_similar.append([i,avg_value])

    most_similar.sort(key=lambda x: x[1],reverse=True)
    return most_similar[:n]
        
def get_vector_list(model, key, df_with_keys=None):
    vector_list = []
    for value in get_key_values(key, True, prefix="key", df_with_keys=df_with_keys):
        vector_list.append(model.wv[value])
        
    return vector_list
    
def evaluate_most_similar_proximity_avg(model, test_cases, verbose, df_with_keys=None):
    most_similar = []
    result = "\n"
    for test_case in test_cases:
        result_text = f"Test für Key_{test_case}: \t"
        result += result_text
        if verbose:
            print(result_text)
        most_similar_sentences = find_n_most_similar_to_key(model, test_case, "proximity_avg", df_with_keys=df_with_keys)
        sentence_of_testcase = get_key_values(test_case, True, prefix="key", df_with_keys=df_with_keys)
        temp = 0
        for sentence in most_similar_sentences:
            output, anzahl_paare = compare_lists(get_key_values(sentence[0], True, prefix="key", df_with_keys=df_with_keys), sentence_of_testcase)
            temp += anzahl_paare
            if verbose:
                print(f"Key_{sentence[0]}: {output}")
        result += f"{temp} gleiche Werte von 55.\n"
    return result

def evaluate_most_similar_proximity_topn_avg(model, test_cases, verbose, df_with_keys=None, method="proximity_topn_avg"):
    most_similar = []
    result = "\n"
    for test_case in test_cases:
        result_text = f"Test für Key_{test_case}: \t"
        result += result_text
        if verbose:
            print(result_text)
        most_similar_sentences = find_n_most_similar_to_key(model, test_case, method, df_with_keys=df_with_keys)
        sentence_of_testcase = get_key_values(test_case, True, prefix="key", df_with_keys=df_with_keys)
        temp = 0
        for sentence in most_similar_sentences:
            output, anzahl_paare = compare_lists(get_key_values(sentence[0], True, prefix="key", df_with_keys=df_with_keys), sentence_of_testcase)
            temp += anzahl_paare
            if verbose:
                print(f"Key_{sentence[0]}: {output}")
        result += f"{temp} gleiche Werte von 55.\n"
    return result

def find_global_most_similar(model, verbose, n=5, df_with_keys=None):
    """
    Vereinfachte Version mit grundlegender Fortschrittsanzeige
    """
    all_similarities = []
    total_cases = len(df_with_keys)
    total_iterations = total_cases * (total_cases - 1) // 2
    iterations_done = 0
    
    start_time = time.time()
    
    # Pre-load vectors
    all_vectors = [get_vector_list(model, i, df_with_keys=df_with_keys) for i in range(total_cases)]
    
    for i in range(total_cases):
        for j in range(i + 1, total_cases):
            similarity = proximity_avg(all_vectors[i], all_vectors[j])
            all_similarities.append((i, j, similarity))
            
            iterations_done += 1
            
            if iterations_done % 1000000 == 0:
                elapsed = time.time() - start_time
                progress = iterations_done / total_iterations
                remaining = (elapsed / progress) - elapsed if progress > 0 else 0
                
                print(f"{iterations_done}/{total_iterations} - "
                      f"Vergangen: {timedelta(seconds=int(elapsed))} - "
                      f"Verbleibend: ~{timedelta(seconds=int(remaining))}")
    
    all_similarities.sort(key=lambda x: x[2], reverse=True)
    return all_similarities[:n]

    
def run_full_vector_test(model_list, verbose, very_verbose=False, df_with_keys=None):
    if verbose:
        print("\n################################################")
        print("             PROXIMITY AVERAGE TEST")
        print("################################################\n")
    model = model_list[0]
    success = {}

    # 1. Most Similar to Key
    success_text = evaluate_most_similar_proximity_avg(model, [
        1,5,500,9999
    ], verbose=verbose, df_with_keys=df_with_keys)
    success[f"Most similar to Key:"] = success_text
    if very_verbose:
        # 2. Most Similar pair
        success_text = find_global_most_similar(model, verbose=verbose, df_with_keys=df_with_keys)
        success[f"Most similar:"] = success_text

    if verbose:
        for x, y in success.items():
            print(x, y)

    if verbose:
        print("\n################################################")
        print("           PROXIMITY TOPN AVERAGE TEST")
        print("################################################\n")
    success = {}
    success_text = evaluate_most_similar_proximity_topn_avg(model, [
        1,5,500,9999
    ], verbose=verbose, df_with_keys=df_with_keys)
    success[f"Most similar to Key (topn):"] = success_text
    
    if verbose:
        for x, y in success.items():
            print(x, y)
    if verbose:
        print("\n################################################")
        print("           PROXIMITY SUBSET PROXIMITY TEST")
        print("################################################\n")
    success = {}
    success_text = evaluate_most_similar_proximity_topn_avg(model, [
        1,5,500,9999
    ], verbose=verbose, df_with_keys=df_with_keys, method="subset_proximity_avg")
    success[f"Most similar to Key (topn):"] = success_text
    if verbose:
        for x, y in success.items():
            print(x, y)
    if verbose:
        print("\n################################################")
        print("           PROXIMITY AVERAGE ALL TEST")
        print("################################################\n")
    success = {}
    success_text = evaluate_most_similar_proximity_topn_avg(model, [
        1,5,500,9999
    ], verbose=verbose, df_with_keys=df_with_keys, method="proximity_avg_all")
    success[f"Most similar to Key (topn):"] = success_text

    if verbose:
        for x, y in success.items():
            print(x, y)