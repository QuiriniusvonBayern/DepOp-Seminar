import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import time
from gensim.models import Word2Vec

@dataclass(frozen=True)
class ExperimentConfig:
    vector_sizes: List[int]
    window_sizes: List[int]
    epochs: List[int]
    negatives: List[int]
    samples: List[float]
    hs_values: List[int]
    alphas: List[float]
    seeds: List[int]


@dataclass(frozen=True)
class ModelConfig:
    """Immutable configuration vor Word2Vec-Models"""
    vector_size:int = 150
    window:int = 4
    epochs:int = 30
    min_count:int = 1
    negative:int = 5
    sample:float = 1e-3
    hs:int = 0
    alpha:float = 0.025
    seed:int = 1
    num_sentences:int = 10000
    workers:int = 4
    sg:int = 0

    # Experiment-CONFIG (VOR Parameter-Sweeps)
    VECTOR_SIZES: List[int] = field(default_factory=lambda: [
        20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 
        85, 90, 95, 100, 110, 120, 130, 140, 150, 180, 225, 
        250, 275, 300
    ])
    
    WINDOW_SIZES: List[int] = field(default_factory=lambda: [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 20
    ])
    
    NUM_EPOCHS: List[int] = field(default_factory=lambda: [
        3, 5, 7, 10, 13, 15, 17, 20, 23, 25, 28, 30, 33, 35,
        38, 40, 43, 45, 48, 50, 55, 60, 65, 70, 75, 80, 90,
        100, 110, 120, 130, 140, 150, 175, 200, 250
    ])
    
    NEGATIVES: List[int] = field(default_factory=lambda: [5, 10])
    SAMPLES: List[float] = field(default_factory=lambda: [1e-3, 1e-4])
    HS_VALUES: List[int] = field(default_factory=lambda: [0, 1])
    ALPHAS: List[float] = field(default_factory=lambda: [0.025, 0.05])
    SEEDS: List[int] = field(default_factory=lambda: [1, 42])
    SENTENCES: List[List[str]] = field(default_factory=list)

class ModelTrainer:
    def train_model(
        self,
        sentences: List[List[str]],
        config: ModelConfig
    ) -> Word2Vec:
        return Word2Vec(
            sentences=sentences,
            vector_size=config.vector_size,
            window=config.window,
            epochs=config.epochs,
            min_count=config.min_count,
            negative=config.negative,
            sample=config.sample,
            hs=config.hs,
            alpha=config.alpha,
            seed=config.seed,
            workers=config.workers,
            sg=config.sg
        )

class ModelRepository:
    def save(self, model, config):

        sentence_length = len(sentences_selection[0])
        alg = "CBOW" if params["sg"] == 0 else "SGNS"

        name = self.get_model_path(
            alg=alg,
            num_sentences=num_sentences,
            sentence_length=sentence_length,
            vector_size=config.vector_size,
            window=config.window,
            epochs=config.epochs,
            min_count=config.min_count,
            negative=config.negative,
            sample=config.sample,
            hs=config.hs,
            alpha=config.alpha,
            seed=config.seed,
            workers=config.workers,
            sg=config.sg
            base_dir=base_dir
        )

    def load(config):
        pass

    def get_model_path(
        self, 
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

class ExperimentRunner:
    """Erzeugt ModelConfig-Varianten

    Ruft ModelTrainer auf

    Nutzt ModelRepository

    Kennt Sweep-Logik"""
    pass



class ModelGenerator:

    #class initiation
    def __init__(self):
        self.base_params = {  
            "vector_size": 150,
            "window": 4,
            "epochs": 30,
            "min_count": 1,
            "negative": 5,
            "sample": 1e-3,
            "hs": 0,
            "alpha": 0.025,
            "seed": 1,
            "num_sentences": 10000,
            "sg": 0,
            "workers": 4,
            "sentences":[]
        }

    def set_base_values(self, base_params):
        self.base_params = base_params
        return

    # Word2Vec trainieren
    def get_model(
        self,
        sentences_tokenized,
        params,
        num_sentences,
        base_dir=None,
        verbose=False
    ):
        sentences_selection = sentences_tokenized[:num_sentences]
        if not sentences_selection:
            raise ValueError("Keine Sätze zum Trainieren übergeben")

        sentence_length = len(sentences_selection[0])
        alg = "CBOW" if params["sg"] == 0 else "SGNS"

        name = self.get_model_path(
            alg=alg,
            vector_size=params["vector_size"],
            window=params["window"],
            epochs=params["epochs"],
            num_sentences=num_sentences,
            sentence_length=sentence_length,
            min_count=params["min_count"],
            negative=params["negative"],
            sample=params["sample"],
            hs=params["hs"],
            alpha=params["alpha"],
            seed=params["seed"],
            base_dir=base_dir
        )
        if verbose:
            print("Started Model-Creation with name: " + str(name))

        if not os.path.exists(name):
            startzeit = time.time()
            if verbose:
                print("Model has to be created new.")
                print("Startzeit: " + str(time.ctime(startzeit)))
            del params["sentences"]
            model = Word2Vec(
                sentences=sentences_selection,
                **params
            )

            model.save(str(name))

            endzeit = time.time()
            verstrichene_zeit = endzeit - startzeit
            minuten, sekunden = divmod(verstrichene_zeit, 60)
            if verbose:
                print("Fertig um: " + str(time.ctime(endzeit)))
                print(f"Nach: {int(minuten)} Minuten und {int(sekunden)} Sekunden")
                print(f"created new Model.")

        else:
            model = Word2Vec.load(str(name))
            if verbose:
                print(f"loaded model.")

        return [
            model,
            {
                "Vector Size": params["vector_size"],
                "Window": params["window"],
                "Algorithmus": alg,
                "Kleinste Worthäufigkeit": params["min_count"],
                "Workers": params["workers"],
                "Epochs": params["epochs"],
                "Satzanzahl": num_sentences,
                "Satzlänge": sentence_length,
                "Negative Sampling": params["negative"],
                "Sample": params["sample"],
                "Hierarchical Softmax": params["hs"],
                "Alpha": params["alpha"],
                "Seed": params["seed"]
            }
        ]

        

    def get_model_path(
        self, 
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


    def create_parameter_sweep_models(self, num_sentences, sweep_values, sweep_param, base_params, base_dir=None, verbose=False):
        if verbose:
            print("...Creating sentence models...")
        models = []
        for value in sweep_values:
            for alg in [0, 1]:
                params = base_params.copy()
                params[sweep_param] = value
                params["sg"] = alg
                models.append(
                    self.get_model(
                        sentences_tokenized=base_params["sentences"],
                        params=params,
                        num_sentences=num_sentences,
                        base_dir=base_dir,
                        verbose=verbose
                    )
                )
        if verbose:
            print("...Finished Creating sentence models...")
        return models

    def create_many_models(
        self,
        models,
        n_sentences_tokenized,
        sentences_tokenized,
        n_used_sentences=10000,
        vector_sizes = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 110, 120, 130, 140, 150, 180, 225, 250, 275, 300],
        window_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20],
        num_epochs = [3, 5, 7, 10, 13, 15, 17, 20, 23, 25, 28, 30, 33, 35, 38, 40, 43, 45, 48, 50, 55, 60, 65, 70, 75, 80, 90, 100, 110, 120, 130, 140, 150, 175, 200, 250],
        negatives=[5, 10],
        samples=[1e-3, 1e-4],
        hs_values=[0, 1],
        alphas=[0.025, 0.05],
        seeds=[1, 42],
        base_dir=None,
        verbose=False
    ):
        self.base_params["sentences"] = sentences_tokenized
        experiments = {
            "sentence": ("sentences", list(n_sentences_tokenized.values())),
            "vector": ("vector_size", vector_sizes),
            "windowsize": ("window", window_sizes),
            "epoch": ("epochs", num_epochs),
            "negative": ("negative", negatives),
            "sample": ("sample", samples),
            "hs": ("hs", hs_values),
            "alpha": ("alpha", alphas),
            "seed": ("seed", seeds),
        }
        
        print("Model Creation started.")
        for name, (param, values) in experiments.items():
            if verbose:
                print(f"...Creating {name} models...")

            models[name] = self.create_parameter_sweep_models(
                base_params=self.base_params,
                sweep_param=param,
                sweep_values=values,
                num_sentences=n_used_sentences,
                base_dir=base_dir,
                verbose=verbose
            )

            if verbose:
                print(f"...Finished Creating {name} models...")

        categories = list(models.keys())
        print("Model Creation finished.")
        return models, categories
    