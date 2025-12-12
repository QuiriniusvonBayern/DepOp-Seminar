import pandas as pd
import numpy as np
import os
from .cosine_tests import get_vector_list
from .key_vector_tests import get_key_values, has_churned
from gensim.models import Word2Vec

def calculate_average_sentence_vectors(model: Word2Vec, df_with_keys: pd.DataFrame) -> pd.Series:
    """
    Calculates the average Vectors of a Sentence in a WOrd2Vec Model

    Prameters:
    ----------
    model : 
        Word2Vec Model created from the sentences from the Dataframe
    df_with_keys : pd.DataFrame
        DataFrame containing the sentences used to train the model
        
    Returns:
    --------
    pd.Series
        Series with average sentence vectors for each key in the DataFrame
    """
    average_sentence_vectors = {}
    churn_status = {}
    indices = {}
    for i in range(len(df_with_keys)):
        key = f"Key_{i+1}"
        sentence_vectors = get_vector_list(model, i, df_with_keys=df_with_keys)
        if len(sentence_vectors) > 0:
            average_sentence_vectors[key] = np.mean(sentence_vectors, axis=0)
            churn_status[key] = has_churned(customer_index=i, df=df_with_keys)
            indices[key] = i
    return pd.DataFrame({
        'index': list(indices.values()), 
        'key': list(average_sentence_vectors.keys()),
        'vector': list(average_sentence_vectors.values()),
        'churned': list(churn_status.values())
    })

def create_average_sentence_vectors(model_list, df_with_keys, base_dir, verbose=False):
    """
    Creates or loads precalculated average sentence vectors for a given Word2Vec model and DataFrame.
    Prameters:
    ----------
    model_list : 
        List containing the Word2Vec Model and a dictionary with model information
    df_with_keys : pd.DataFrame
        DataFrame containing the sentences used to train the model
    base_dir : str
        Base directory from which the script is run, used to construct the file path for saving/loading
    verbose : bool, optional
        If True, prints status messages (default is False)
    Returns:
    --------
    pd.Series
        Series with average sentence vectors for each key in the DataFrame 
    """
    name = f"{base_dir}/precalculated_values/average_sentence_vectors_"
    model = model_list[0]
    model_dict = model_list[1]
    for key, value in model_dict.items():
        name += f"_{key[:3]}_{value}"
    name += ".npz" 

    if not os.path.exists(name):
        if verbose:
            print(f"Creating average sentence vectors and saving to {name}")
        vectors= calculate_average_sentence_vectors(model, df_with_keys)
        
        np.savez(name, index=vectors['index'].to_numpy(), key=vectors['key'].to_numpy(), vector=np.array(vectors['vector'].tolist()),  churned=vectors['churned'].to_numpy())
        return vectors  
    else:
        if verbose:
            print(f"Reading average sentence vectors from {name}")
        array_data = np.load(name, allow_pickle=True)
        df = pd.DataFrame({
            'index': array_data['index'],
            'key': array_data['key'],
            'vector': list(array_data['vector']),
            'churned': array_data['churned']
        })
        df = df.sort_values('index').reset_index(drop=True)
        assert df['index'].is_monotonic_increasing, "Index ist nicht sortiert!"
        assert len(df) == df['index'].max() + 1, "Index hat Lücken!"
        
        return df