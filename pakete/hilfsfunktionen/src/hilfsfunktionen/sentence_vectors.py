import pandas as pd
import numpy as np
import os
from .benchmark_of_semantic_operatiors import get_vector_list
from .key_vector_tests import get_key_values, has_churned, is_female, in_country
from gensim.models import Word2Vec


def calculate_average_sentence_vectors(
    model: Word2Vec, df_with_keys: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate the average Word2Vec vector for each sentence in the DataFrame.

    For each row in the DataFrame, the function retrieves the word vectors of the
    corresponding sentence and computes their element-wise mean. Additional metadata
    (churn status, gender, country) is also collected for each entry.

    Parameters
    ----------
    model : Word2Vec
        Trained Word2Vec model used to obtain word vectors.
    df_with_keys : pd.DataFrame
        DataFrame containing the sentences used to train the model.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - index: original row index from the input DataFrame
        - key: unique identifier for each sentence (Key_<index>)
        - vector: average sentence vector as a numpy array
        - churned: churn status of the corresponding customer
        - female: gender indicator of the corresponding customer
        - country: country of the corresponding customer
    """
    indices = []
    keys = []
    vectors = []
    churned_status = []
    female_status = []
    country_status = []

    for i in range(len(df_with_keys)):
        key = f"Key_{i+1}"
        sentence_vectors = get_vector_list(model, i, df_with_keys=df_with_keys)

        if len(sentence_vectors) > 0:
            indices.append(i)
            keys.append(key)
            vectors.append(np.mean(sentence_vectors, axis=0))
            churned_status.append(has_churned(customer_index=i, df=df_with_keys))
            female_status.append(is_female(customer_index=i, df=df_with_keys))
            country_status.append(in_country(customer_index=i, df=df_with_keys))

    return pd.DataFrame(
        {
            "index": indices,
            "key": keys,
            "vector": vectors,
            "churned": churned_status,
            "female": female_status,
            "country": country_status,
        }
    )


def create_average_sentence_vectors(
    model_list, df_with_keys: pd.DataFrame, base_dir: str, verbose: bool = False
) -> pd.DataFrame:
    """
    Load precomputed average sentence vectors from disk or compute and save them.

    The function constructs a file name based on model parameters. If the file
    already exists, it loads the vectors and verifies index integrity. Otherwise,
    it computes the vectors using calculate_average_sentence_vectors and saves them
    as a compressed .npz file.

    Parameters
    ----------
    model_list : list
        A list containing the Word2Vec model and a dictionary with model parameters.
        Expected format: [model, model_dict]
    df_with_keys : pd.DataFrame
        DataFrame containing the sentences used to train the model.
    base_dir : str
        Base directory for constructing the file path to save/load vectors.
    verbose : bool, optional
        If True, print status messages (default is False).

    Returns
    -------
    pd.DataFrame
        DataFrame with average sentence vectors and associated metadata.
        Same structure as returned by calculate_average_sentence_vectors.
    """
    model = model_list[0]
    model_dict = model_list[1]

    file_name = f"{base_dir}/precalculated_values/average_sentence_vectors_"
    for key, value in model_dict.items():
        file_name += f"_{key[:3]}_{value}"
    file_name += ".npz"

    if not os.path.exists(file_name):
        if verbose:
            print(f"Creating average sentence vectors and saving to {file_name}")

        vectors_df = calculate_average_sentence_vectors(model, df_with_keys)

        np.savez(
            file_name,
            index=vectors_df["index"].to_numpy(),
            key=vectors_df["key"].to_numpy(),
            vector=np.array(vectors_df["vector"].tolist()),
            churned=vectors_df["churned"].to_numpy(),
            female=vectors_df["female"].to_numpy(),
            country=vectors_df["country"].to_numpy(),
        )
        return vectors_df
    else:
        if verbose:
            print(f"Reading average sentence vectors from {file_name}")

        loaded_data = np.load(file_name, allow_pickle=True)
        vectors_df = pd.DataFrame(
            {
                "index": loaded_data["index"],
                "key": loaded_data["key"],
                "vector": list(loaded_data["vector"]),
                "churned": loaded_data["churned"],
                "female": loaded_data["female"],
                "country": loaded_data["country"],
            }
        )

        vectors_df = vectors_df.sort_values("index").reset_index(drop=True)
        assert (
            vectors_df["index"].is_monotonic_increasing
        ), "Index is not monotonically increasing!"
        assert (
            len(vectors_df) == vectors_df["index"].max() + 1
        ), "Index has gaps!"

        return vectors_df
