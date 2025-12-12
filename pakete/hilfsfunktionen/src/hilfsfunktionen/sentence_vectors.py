import pandas as pd
import numpy as np
import os
from .cosine_tests import get_vector_list
from .key_vector_tests import get_key_values

def calculate_average_sentence_vectors(model, df_with_keys):
    average_sentence_vectors = {}
    for i in range(len(df_with_keys)):
        average_sentence_vectors[f"Key_{i}"] = []
        sentence_vectors = get_vector_list(model, i, df_with_keys=df_with_keys)
        average_sentence_vectors[f"Key_{i}"].append(np.mean(sentence_vectors, axis=0))
    return pd.Series(average_sentence_vectors, copy=False)

def create_average_sentence_vectors(model_list, df_with_keys, base_dir, verbose=False):
    name = f"{base_dir}/precalculated_values/average_sentence_vectors_"
    model = model_list[0]
    model_dict = model_list[1]
    for key, value in model_dict.items():
        name += f"_{key[:3]}_{value}"
    name += ".npz" 

    if not os.path.exists(name):
        if verbose:
            print(f"Creating average sentence vectors and saving to {name}")
        vectors = calculate_average_sentence_vectors(model, df_with_keys)
        words = list(vectors.keys())  # Reihenfolge bleibt
        embeddings = np.array([vectors[word] for word in words])
        np.savez(name, words=words, embeddings=embeddings)
        return vectors  
    else:
        if verbose:
            print(f"Reading average sentence vectors from {name}")
        array_data = np.load(name, allow_pickle=False)
        words = array_data['words']
        embeddings = array_data['embeddings']
        vectors = {}
        for i, word in enumerate(words):
            vectors[word] = embeddings[i] 
        return pd.Series(vectors, copy=False)