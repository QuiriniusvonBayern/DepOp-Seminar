import os
from pathlib import Path
from gensim.test.utils import common_texts
from itertools import combinations
from numpy.linalg import norm
from gensim.models import Word2Vec
import time
from datetime import timedelta

# Word2Vec trainieren
def get_model(
    sentences_tokenized,
    vector_size,
    window,
    Skip_Gram,
    min_count,
    workers,
    epochs,
    num_sentences,
    negative,
    sample,
    hs,
    alpha,
    seed,
    base_dir=None,
    verbose=False
):
    sentences_selection = sentences_tokenized[:num_sentences]
    sentence_length = len(sentences_selection[0])

    alg = "CBOW"
    if Skip_Gram == 1:
        alg = "SGNS"

    name = get_model_path(
        alg,
        vector_size,
        window,
        epochs,
        num_sentences,
        sentence_length,
        min_count,
        negative,
        sample,
        hs,
        alpha,
        seed,
        base_dir
    )

    if verbose:
        print("Started Model-Creation with name: " + str(name))

    if not os.path.exists(name):
        startzeit = time.time()
        if verbose:
            print("Model has to be created new.")
            print("Startzeit: " + str(time.ctime(startzeit)))

        model = Word2Vec(
            sentences=sentences_selection,
            vector_size=vector_size,
            window=window,
            sg=Skip_Gram,
            min_count=min_count,
            workers=workers,
            epochs=epochs,
            negative=negative,
            sample=sample,
            hs=hs,
            alpha=alpha,
            seed=seed
        )

        model.save(str(name))

        endzeit = time.time()
        verstrichene_zeit = endzeit - startzeit
        if verbose:
            print("Fertig um: " + str(time.ctime(endzeit)))
            print(f"Nach: {verstrichene_zeit:.4f} Sekunden")
            print(f"created new Model.")

    else:
        model = Word2Vec.load(str(name))
        if verbose:
            print(f"loaded model.")

    return [
        model,
        {
            "Vector Size": vector_size,
            "Window": window,
            "Algorithmus": alg,
            "Kleinste Worthäufigkeit": min_count,
            "Workers": workers,
            "Epochs": epochs,
            "Satzanzahl": num_sentences,
            "Satzlänge": sentence_length,
            "Negative Sampling": negative,
            "Sample": sample,
            "Hierarchical Softmax": hs,
            "Alpha": alpha,
            "Seed": seed
        }
    ]

    

def get_model_path(
    alg,
    vector_size,
    window,
    epochs,
    num_sentences,
    sentence_length,
    min_count,
    negative,
    sample,
    hs,
    alpha,
    seed,
    base_dir=None
):
    """
    Erstellt eindeutigen Modell-Dateinamen basierend auf allen variablen Parametern.
    """

    from pathlib import Path

    # Erweiterter eindeutiger Dateiname
    filename = (
        f"{alg}"
        f"_vs_{vector_size}"
        f"_win_{window}"
        f"_ep_{epochs}"
        f"_nsent_{num_sentences}"
        f"_sentlen_{sentence_length}"
        f"_mc_{min_count}"
        f"_neg_{negative}"
        f"_sample_{sample}"
        f"_hs_{hs}"
        f"_alpha_{alpha}"
        f"_seed_{seed}"
        ".model"
    )

    # Basisordner automatisch erkennen
    if base_dir is None:
        base_dir = Path.cwd()

        for parent in base_dir.parents:
            if (parent / "models").exists():
                base_dir = parent
                break
    else:
        base_dir = Path(base_dir)

    model_path = base_dir / "models" / filename
    model_path.parent.mkdir(exist_ok=True, parents=True)

    return model_path


def create_comparison_models(sentences_tokenized, base_params, base_dir=None, verbose=False):
    if verbose:
        print("...Creating comparison model...")
    models = []
    for alg in [0, 1]:
        models.append(
            get_model(
                sentences_tokenized=sentences_tokenized,
                vector_size=base_params["vector_size"],
                window=base_params["window"],
                Skip_Gram=alg,
                min_count=base_params["min_count"],
                workers=4,
                epochs=base_params["epochs"],
                num_sentences=base_params["num_sentences"],
                negative=base_params["negative"],
                sample=base_params["sample"],
                hs=base_params["hs"],
                alpha=base_params["alpha"],
                seed=base_params["seed"],
                base_dir=base_dir,
                verbose=verbose
            )
        )
    if verbose:
        print("...Finished Creating comparison model...")
    return models


def create_sentence_models(n_sentences_tokenized, base_params, base_dir=None, verbose=False):
    if verbose:
        print("...Creating sentence models...")
    models = []
    for sentences in n_sentences_tokenized.values():
        for alg in [0, 1]:
            models.append(
                get_model(
                    sentences_tokenized=sentences,
                    vector_size=base_params["vector_size"],
                    window=base_params["window"],
                    Skip_Gram=alg,
                    min_count=base_params["min_count"],
                    workers=4,
                    epochs=base_params["epochs"],
                    num_sentences=base_params["num_sentences"],
                    negative=base_params["negative"],
                    sample=base_params["sample"],
                    hs=base_params["hs"],
                    alpha=base_params["alpha"],
                    seed=base_params["seed"],
                    base_dir=base_dir,
                    verbose=verbose
                )
            )
    if verbose:
        print("...Finished Creating sentence models...")
    return models


def create_vector_models(sentences_tokenized, base_params, vector_sizes, base_dir=None, verbose=False):
    if verbose:
        print("...Creating vector size models...")
    models = []
    for v in vector_sizes:
        for alg in [0, 1]:
            models.append(
                get_model(
                    sentences_tokenized=sentences_tokenized,
                    vector_size=v,
                    window=base_params["window"],
                    Skip_Gram=alg,
                    min_count=base_params["min_count"],
                    workers=4,
                    epochs=base_params["epochs"],
                    num_sentences=base_params["num_sentences"],
                    negative=base_params["negative"],
                    sample=base_params["sample"],
                    hs=base_params["hs"],
                    alpha=base_params["alpha"],
                    seed=base_params["seed"],
                    base_dir=base_dir,
                    verbose=verbose
                )
            )
    if verbose:
        print("...Finished Creating vector size models...")
    return models


def create_window_models(sentences_tokenized, base_params, window_sizes, base_dir=None, verbose=False):
    if verbose:
        print("...Creating window size models...")
    models = []
    for w in window_sizes:
        for alg in [0, 1]:
            models.append(
                get_model(
                    sentences_tokenized=sentences_tokenized,
                    vector_size=base_params["vector_size"],
                    window=w,
                    Skip_Gram=alg,
                    min_count=base_params["min_count"],
                    workers=4,
                    epochs=base_params["epochs"],
                    num_sentences=base_params["num_sentences"],
                    negative=base_params["negative"],
                    sample=base_params["sample"],
                    hs=base_params["hs"],
                    alpha=base_params["alpha"],
                    seed=base_params["seed"],
                    base_dir=base_dir,
                    verbose=verbose
                )
            )
    if verbose:
        print("...Finished Creating window size models...")
    return models


def create_epoch_models(sentences_tokenized, base_params, num_epochs, base_dir=None, verbose=False):
    if verbose:
        print("...Creating epoch models...")
    models = []
    for e in num_epochs:
        for alg in [0, 1]:
            models.append(
                get_model(
                    sentences_tokenized=sentences_tokenized,
                    vector_size=base_params["vector_size"],
                    window=base_params["window"],
                    Skip_Gram=alg,
                    min_count=base_params["min_count"],
                    workers=4,
                    epochs=e,
                    num_sentences=base_params["num_sentences"],
                    negative=base_params["negative"],
                    sample=base_params["sample"],
                    hs=base_params["hs"],
                    alpha=base_params["alpha"],
                    seed=base_params["seed"],
                    base_dir=base_dir,
                    verbose=verbose
                )
            )
    if verbose:
        print("...Finished Creating epoch models...")
    return models


def create_min_count_models(sentences_tokenized, base_params, min_counts, base_dir=None, verbose=False):
    if verbose:
        print("...Creating min count models...")
    models = []
    for mc in min_counts:
        for alg in [0, 1]:
            models.append(
                get_model(
                    sentences_tokenized=sentences_tokenized,
                    vector_size=base_params["vector_size"],
                    window=base_params["window"],
                    Skip_Gram=alg,
                    min_count=mc,
                    workers=4,
                    epochs=base_params["epochs"],
                    num_sentences=base_params["num_sentences"],
                    negative=base_params["negative"],
                    sample=base_params["sample"],
                    hs=base_params["hs"],
                    alpha=base_params["alpha"],
                    seed=base_params["seed"],
                    base_dir=base_dir,
                    verbose=verbose
                )
            )
    if verbose:
        print("...Finished Creating min count models...")
    return models


def create_negative_models(sentences_tokenized, base_params, negatives, base_dir=None, verbose=False):
    if verbose:
        print("...Creating negative sampling models...")
    models = []
    for n in negatives:
        for alg in [0, 1]:
            models.append(
                get_model(
                    sentences_tokenized=sentences_tokenized,
                    vector_size=base_params["vector_size"],
                    window=base_params["window"],
                    Skip_Gram=alg,
                    min_count=base_params["min_count"],
                    workers=4,
                    epochs=base_params["epochs"],
                    num_sentences=base_params["num_sentences"],
                    negative=n,
                    sample=base_params["sample"],
                    hs=base_params["hs"],
                    alpha=base_params["alpha"],
                    seed=base_params["seed"],
                    base_dir=base_dir,
                    verbose=verbose
                )
            )
    if verbose:
        print("...Finished Creating negative sampling models...")
    return models


def create_sample_models(sentences_tokenized, base_params, samples, base_dir=None, verbose=False):
    if verbose:
        print("...Creating sample models...")
    models = []
    for s in samples:
        for alg in [0, 1]:
            models.append(
                get_model(
                    sentences_tokenized=sentences_tokenized,
                    vector_size=base_params["vector_size"],
                    window=base_params["window"],
                    Skip_Gram=alg,
                    min_count=base_params["min_count"],
                    workers=4,
                    epochs=base_params["epochs"],
                    num_sentences=base_params["num_sentences"],
                    negative=base_params["negative"],
                    sample=s,
                    hs=base_params["hs"],
                    alpha=base_params["alpha"],
                    seed=base_params["seed"],
                    base_dir=base_dir,
                    verbose=verbose
                )
            )
    if verbose:
        print("...Finished Creating sample models...")
    return models


def create_hs_models(sentences_tokenized, base_params, hs_values, base_dir=None, verbose=False):
    if verbose:
        print("...Creating hierarchical softmax models...")
    models = []
    for h in hs_values:
        for alg in [0, 1]:
            models.append(
                get_model(
                    sentences_tokenized=sentences_tokenized,
                    vector_size=base_params["vector_size"],
                    window=base_params["window"],
                    Skip_Gram=alg,
                    min_count=base_params["min_count"],
                    workers=4,
                    epochs=base_params["epochs"],
                    num_sentences=base_params["num_sentences"],
                    negative=base_params["negative"],
                    sample=base_params["sample"],
                    hs=h,
                    alpha=base_params["alpha"],
                    seed=base_params["seed"],
                    base_dir=base_dir,
                    verbose=verbose
                )
            )
    if verbose:
        print("...Finished Creating hierarchical softmax models...")
    return models


def create_alpha_models(sentences_tokenized, base_params, alphas, base_dir=None, verbose=False):
    if verbose:
        print("...Creating alpha models...")
    models = []
    for a in alphas:
        for alg in [0, 1]:
            models.append(
                get_model(
                    sentences_tokenized=sentences_tokenized,
                    vector_size=base_params["vector_size"],
                    window=base_params["window"],
                    Skip_Gram=alg,
                    min_count=base_params["min_count"],
                    workers=4,
                    epochs=base_params["epochs"],
                    num_sentences=base_params["num_sentences"],
                    negative=base_params["negative"],
                    sample=base_params["sample"],
                    hs=base_params["hs"],
                    alpha=a,
                    seed=base_params["seed"],
                    base_dir=base_dir,
                    verbose=verbose
                )
            )
    if verbose:
        print("...Finished Creating alpha models...")
    return models


def create_seed_models(sentences_tokenized, base_params, seeds, base_dir=None, verbose=False):
    if verbose:
        print("...Creating seed models...")
    models = []
    for sd in seeds:
        for alg in [0, 1]:
            models.append(
                get_model(
                    sentences_tokenized=sentences_tokenized,
                    vector_size=base_params["vector_size"],
                    window=base_params["window"],
                    Skip_Gram=alg,
                    min_count=base_params["min_count"],
                    workers=4,
                    epochs=base_params["epochs"],
                    num_sentences=base_params["num_sentences"],
                    negative=base_params["negative"],
                    sample=base_params["sample"],
                    hs=base_params["hs"],
                    alpha=base_params["alpha"],
                    seed=sd,
                    base_dir=base_dir,
                    verbose=verbose
                )
            )
    if verbose:
        print("...Finished Creating seed models...")
    return models

def create_many_models(
    models,
    n_sentences_tokenized,
    sentences_tokenized,
    vector_sizes=[25, 50, 100, 150, 300, 1000],
    window_sizes=[2, 3, 4, 5, 7, 17],
    num_epochs=[5, 10, 15, 20, 30, 50],
    min_counts=[1, 2, 5],
    negatives=[5, 10],
    samples=[1e-3, 1e-4],
    hs_values=[0, 1],
    alphas=[0.025, 0.05],
    seeds=[1, 42],
    base_dir=None,
    verbose=False
):

    base_params = {
        "vector_size": 150,
        "window": 4,
        "epochs": 30,
        "min_count": 1,
        "negative": 5,
        "sample": 1e-3,
        "hs": 0,
        "alpha": 0.025,
        "seed": 1,
        "num_sentences": 10000
    }
    print("Model Creation started.")
    models["comparison"] = create_comparison_models(sentences_tokenized, base_params, base_dir, verbose)
    models["sentences"] = create_sentence_models(n_sentences_tokenized, base_params, base_dir, verbose)
    models["vector"] = create_vector_models(sentences_tokenized, base_params, vector_sizes, base_dir, verbose)
    models["windowsize"] = create_window_models(sentences_tokenized, base_params, window_sizes, base_dir, verbose)
    models["epoch"] = create_epoch_models(sentences_tokenized, base_params, num_epochs, base_dir, verbose)
    #models["min_count"] = create_min_count_models(sentences_tokenized, base_params, min_counts, base_dir, verbose)
    models["negative"] = create_negative_models(sentences_tokenized, base_params, negatives, base_dir, verbose)
    models["sample"] = create_sample_models(sentences_tokenized, base_params, samples, base_dir, verbose)
    models["hs"] = create_hs_models(sentences_tokenized, base_params, hs_values, base_dir, verbose)
    models["alpha"] = create_alpha_models(sentences_tokenized, base_params, alphas, base_dir, verbose)
    models["seed"] = create_seed_models(sentences_tokenized, base_params, seeds, base_dir, verbose)

    categories = list(models.keys())
    print("Model Creation finished.")
    return models, categories