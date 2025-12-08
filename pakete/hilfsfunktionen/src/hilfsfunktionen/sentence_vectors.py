import pandas as pd
import numpy as np
import os
from hilfsfunktionen.cosine_tests import get_vector_list

def calculate_average_sentence_vectors(model, df_with_keys):
    average_sentence_vectors = []
    for i in range(len(df_with_keys)):
        sentence_vectors = get_vector_list(model, i, df_with_keys=df_with_keys)
        average_sentence_vectors.append(np.mean(sentence_vectors, axis=0))
    return pd.Series(average_sentence_vectors, copy=False)

def create_average_sentence_vectors(model_list, df_with_keys, base_dir, verbose=False):
    name = f"{base_dir}/precalculated_values/average_sentence_vectors_"
    model = model_list[0]
    model_dict = model_list[1]
    for key, value in model_dict.items():
        name += f"_{key[:3]}_{value}"
    name += ".csv"
    if not (os.path.exists(name)):
        if verbose:
            print(f"Creating average sentence vectors and saving to {name}")
        vectors = calculate_average_sentence_vectors(model, df_with_keys)
        vectors.to_csv(name, index=False)
    else:
        if verbose:
            print(f"Reading average sentence vectors from {name}")
        vectors = pd.read_csv(name)
    return vectors
