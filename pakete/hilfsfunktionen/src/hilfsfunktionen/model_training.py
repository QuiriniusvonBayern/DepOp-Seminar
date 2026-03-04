import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import time
from gensim.models import Word2Vec
from itertools import product


@dataclass(frozen=True)
class ModelConfig:
    """Immutable configuration container for Word2Vec model hyperparameters."""

    vector_size: int = 50
    window: int = 4
    epochs: int = 30
    min_count: int = 1
    negative: int = 5
    sample: float = 1e-3
    hs: int = 0
    alpha: float = 0.025
    seed: int = 1
    workers: int = 4
    sg: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary for Word2Vec initialization."""
        return {
            "vector_size": self.vector_size,
            "window": self.window,
            "epochs": self.epochs,
            "min_count": self.min_count,
            "negative": self.negative,
            "sample": self.sample,
            "hs": self.hs,
            "alpha": self.alpha,
            "seed": self.seed,
            "workers": self.workers,
            "sg": self.sg,
        }

    def get_algorithm_name(self) -> str:
        """Return the algorithm name based on the sg parameter."""
        return "CBOW" if self.sg == 0 else "SGNS"


class ModelTrainer:
    """Responsible for training a single Word2Vec model with a given configuration."""

    def train_model(
        self,
        sentences: List[List[str]],
        config: ModelConfig,
        verbose: bool = False,
    ) -> Word2Vec:
        """
        Train a Word2Vec model using the provided sentences and configuration.

        Args:
            sentences: List of tokenized sentences.
            config: ModelConfig containing all hyperparameters.
            verbose: If True, print training progress and timing information.

        Returns:
            A trained Word2Vec model.

        Raises:
            ValueError: If the sentences list is empty.
        """
        if not sentences:
            raise ValueError("No sentences provided for training.")

        if verbose:
            print(f"Training model with {len(sentences)} sentences...")
            start_time = time.time()
            print("Start time: " + str(time.ctime(start_time)))

        model = Word2Vec(sentences=sentences, **config.to_dict())

        if verbose:
            end_time = time.time()
            elapsed_time = end_time - start_time
            minutes, seconds = divmod(elapsed_time, 60)
            print("Finished at: " + str(time.ctime(end_time)))
            print(f"Elapsed time: {int(minutes)} minutes and {int(seconds)} seconds")

        return model


class ModelRepository:
    """Handles saving and loading of Word2Vec models with deterministic file paths."""

    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize the repository with a base directory for model storage.

        Args:
            base_dir: Base directory for models. If None, the directory is auto-detected.
        """
        self.base_dir = self._find_base_dir(base_dir)

    def _find_base_dir(self, base_dir: Optional[Path]) -> Path:
        """Locate or create the base directory for storing models."""
        if base_dir is None:
            base_dir = Path.cwd()
            for parent in base_dir.parents:
                if (parent / "models").exists():
                    base_dir = parent
                    break
        else:
            base_dir = Path(base_dir)
        return base_dir

    def get_model_path(
        self,
        config: ModelConfig,
        num_sentences: int,
        sentence_length: int,
    ) -> Path:
        """
        Generate a unique file path for a model based on its configuration and training data.

        Args:
            config: ModelConfig used for training.
            num_sentences: Number of sentences used for training.
            sentence_length: Length of the sentences used.

        Returns:
            Path object pointing to the model file.
        """
        alg = config.get_algorithm_name()
        filename = (
            f"{alg}"
            f"_vs_{config.vector_size}"
            f"_win_{config.window}"
            f"_ep_{config.epochs}"
            f"_nsent_{num_sentences}"
            f"_sentlen_{sentence_length}"
            f"_mc_{config.min_count}"
            f"_neg_{config.negative}"
            f"_sample_{config.sample}"
            f"_hs_{config.hs}"
            f"_alpha_{config.alpha}"
            f"_seed_{config.seed}"
            ".model"
        )
        model_path = self.base_dir / "models" / filename
        model_path.parent.mkdir(exist_ok=True, parents=True)
        return model_path

    def save(
        self,
        model: Word2Vec,
        config: ModelConfig,
        num_sentences: int,
        sentence_length: int,
        verbose: bool = False,
    ) -> Path:
        """
        Save a trained model to disk.

        Args:
            model: Trained Word2Vec model.
            config: ModelConfig of the model.
            num_sentences: Number of sentences used for training.
            sentence_length: Length of the sentences used.
            verbose: If True, print the save path.

        Returns:
            Path to the saved model file.
        """
        path = self.get_model_path(config, num_sentences, sentence_length)
        model.save(str(path))
        if verbose:
            print(f"Model saved: {path}")
        return path

    def load(
        self,
        config: ModelConfig,
        num_sentences: int,
        sentence_length: int,
        verbose: bool = False,
    ) -> Optional[Word2Vec]:
        """
        Load a model from disk if it exists.

        Args:
            config: ModelConfig of the desired model.
            num_sentences: Number of sentences used for training.
            sentence_length: Length of the sentences used.
            verbose: If True, print the load path.

        Returns:
            The loaded Word2Vec model, or None if the file does not exist.
        """
        path = self.get_model_path(config, num_sentences, sentence_length)
        if not path.exists():
            return None
        model = Word2Vec.load(str(path))
        if verbose:
            print(f"Model loaded: {path}")
        return model

    def exists(
        self, config: ModelConfig, num_sentences: int, sentence_length: int
    ) -> bool:
        """Check if a model file exists for the given configuration."""
        path = self.get_model_path(config, num_sentences, sentence_length)
        return path.exists()


class ExperimentRunner:
    """Orchestrates multiple training runs for parameter sweeps and grid searches."""

    def __init__(
        self,
        trainer: ModelTrainer,
        repository: ModelRepository,
        base_config: ModelConfig,
    ):
        """
        Initialize the experiment runner.

        Args:
            trainer: ModelTrainer instance.
            repository: ModelRepository instance.
            base_config: Base configuration for experiments.
        """
        self.trainer = trainer
        self.repository = repository
        self.base_config = base_config

    def run_single_experiment(
        self,
        sentences: List[List[str]],
        config: ModelConfig,
        num_sentences: int,
        verbose: bool = False,
    ) -> Tuple[Word2Vec, Dict[str, Any]]:
        """
        Run a single experiment: train or load a model and return it with metadata.

        Args:
            sentences: Tokenized sentences.
            config: ModelConfig for this experiment.
            num_sentences: Number of sentences to use from the beginning of the list.
            verbose: If True, print status messages.

        Returns:
            A tuple containing the trained/loaded model and its metadata.
        """
        sentences_selection = sentences[:num_sentences]
        sentence_length = len(sentences_selection[0]) if sentences_selection else 0

        if verbose:
            print(
                f"Started model creation with: {config.get_algorithm_name()}, "
                f"vs={config.vector_size}, win={config.window}, ep={config.epochs}"
            )

        model = self.repository.load(config, num_sentences, sentence_length, verbose)

        if model is None:
            if verbose:
                print("Model not found; training new model.")
            model = self.trainer.train_model(sentences_selection, config, verbose)
            self.repository.save(model, config, num_sentences, sentence_length, verbose)
            if verbose:
                print("New model created and saved.")
        else:
            if verbose:
                print("Model loaded from disk.")

        metadata = {
            "Vector Size": config.vector_size,
            "Window": config.window,
            "Algorithmus": config.get_algorithm_name(),
            "Kleinste Worthäufigkeit": config.min_count,
            "Workers": config.workers,
            "Epochs": config.epochs,
            "Satzanzahl": num_sentences,
            "Satzlänge": sentence_length,
            "Negative Sampling": config.negative,
            "Sample": config.sample,
            "Hierarchical Softmax": config.hs,
            "Alpha": config.alpha,
            "Seed": config.seed,
        }
        return model, metadata

    def create_parameter_sweep_models(
        self,
        sentences: List[List[str]],
        sweep_param: str,
        sweep_values: List[Any],
        num_sentences: int,
        verbose: bool = False,
    ) -> List[Tuple[Word2Vec, Dict[str, Any]]]:
        """
        Create models by varying a single parameter across specified values.

        Args:
            sentences: Tokenized sentences.
            sweep_param: Name of the parameter to vary.
            sweep_values: List of values for the parameter.
            num_sentences: Number of sentences to use.
            verbose: If True, print status messages.

        Returns:
            List of (model, metadata) tuples for each combination.
        """
        if verbose:
            print(f"...Creating {sweep_param} models...")

        models = []
        for value in sweep_values:
            for alg in [0, 1]:
                config_dict = self.base_config.to_dict()
                config_dict[sweep_param] = value
                config_dict["sg"] = alg
                config = ModelConfig(**config_dict)

                model, metadata = self.run_single_experiment(
                    sentences=sentences,
                    config=config,
                    num_sentences=num_sentences,
                    verbose=verbose,
                )
                models.append((model, metadata))

        if verbose:
            print(f"...Finished creating {sweep_param} models...")
        return models

    def run_grid_search(
        self,
        sentences: List[List[str]],
        param_grid: Dict[str, List[Any]],
        num_sentences: int,
        verbose: bool = False,
    ) -> List[Tuple[Word2Vec, Dict[str, Any]]]:
        """
        Perform a grid search over the given parameter grid.

        Args:
            sentences: Tokenized sentences.
            param_grid: Dictionary mapping parameter names to lists of values.
            num_sentences: Number of sentences to use.
            verbose: If True, print status messages.

        Returns:
            List of (model, metadata) tuples for all combinations.
        """
        results = []
        grid_keys = list(param_grid.keys())
        grid_values = list(param_grid.values())

        for values in product(*grid_values):
            grid_params = dict(zip(grid_keys, values))
            config_dict = self.base_config.to_dict()
            config_dict.update(grid_params)
            config = ModelConfig(**config_dict)

            model, metadata = self.run_single_experiment(
                sentences=sentences,
                config=config,
                num_sentences=num_sentences,
                verbose=verbose,
            )
            results.append((model, metadata))
        return results

    def create_many_models(
        self,
        sentences: List[List[str]],
        n_sentences_tokenized: Dict[int, List[List[str]]],
        n_used_sentences: int = 10000,
        vector_sizes: List[int] = None,
        window_sizes: List[int] = None,
        num_epochs: List[int] = None,
        negatives: List[int] = None,
        samples: List[float] = None,
        hs_values: List[int] = None,
        alphas: List[float] = None,
        seeds: List[int] = None,
        verbose: bool = False,
    ) -> Tuple[Dict[str, List], List[str]]:
        """
        Generate a collection of models by varying multiple parameters.

        Args:
            sentences: Tokenized sentences.
            n_sentences_tokenized: Dictionary mapping sentence counts to sentence lists.
            n_used_sentences: Default number of sentences to use.
            vector_sizes: List of vector sizes to test.
            window_sizes: List of window sizes to test.
            num_epochs: List of epoch counts to test.
            negatives: List of negative sampling values.
            samples: List of sample values.
            hs_values: List of hierarchical softmax values.
            alphas: List of learning rates.
            seeds: List of random seeds.
            verbose: If True, print status messages.

        Returns:
            A tuple containing a dictionary of models by category and a list of categories.
        """
        if vector_sizes is None:
            vector_sizes = [
                20,
                30,
                40,
                50,
                60,
                80,
                100,
                120,
                140,
                150,
                160,
                180,
                225,
                250,
                275,
                300,
                350,
                400,
            ]
        if window_sizes is None:
            window_sizes = [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                20,
            ]
        if num_epochs is None:
            num_epochs = [
                3,
                5,
                7,
                10,
                13,
                15,
                17,
                20,
                23,
                25,
                28,
                30,
                35,
                40,
                45,
                50,
                60,
                70,
                80,
                90,
                100,
                150,
                200,
            ]
        if negatives is None:
            negatives = [5, 10]
        if samples is None:
            samples = [0, 1e-3, 1e-4]
        if hs_values is None:
            hs_values = [0, 1]
        if alphas is None:
            alphas = [0.025, 0.05]
        if seeds is None:
            seeds = [1, 42]

        experiments = {
            "sentence": ("sentences", list(n_sentences_tokenized.values())),
            "vector": ("vector_size", vector_sizes),
            "windowsize": ("window", window_sizes),
            "epoch": ("epochs", num_epochs),
        }

        models = {}
        print("Model creation started.")

        for name, (param, values) in experiments.items():
            if verbose:
                print(f"...Creating {name} models...")

            if name == "sentence":
                sweep_models = []
                for sent in values:
                    for alg in [0, 1]:
                        config_dict = self.base_config.to_dict()
                        config_dict["sg"] = alg
                        config = ModelConfig(**config_dict)

                        model, metadata = self.run_single_experiment(
                            sentences=sent,
                            config=config,
                            num_sentences=n_used_sentences,
                            verbose=verbose,
                        )
                        sweep_models.append((model, metadata))
                models[name] = sweep_models
            else:
                models[name] = self.create_parameter_sweep_models(
                    sentences=sentences,
                    sweep_param=param,
                    sweep_values=values,
                    num_sentences=n_used_sentences,
                    verbose=verbose,
                )

            if verbose:
                print(f"...Finished creating {name} models...")

        categories = list(models.keys())
        print("Model creation finished.")
        return models, categories


class ModelGenerator:
    """
    Compatibility wrapper for legacy code.
    Delegates to the new modular classes.
    """

    def __init__(self, base_dir: Optional[Path] = None):
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
            "sentences": [],
        }

        base_config = ModelConfig(
            **{k: v for k, v in self.base_params.items() if k not in ["sentences", "num_sentences"]}
        )
        self.trainer = ModelTrainer()
        self.repository = ModelRepository(base_dir)
        self.runner = ExperimentRunner(self.trainer, self.repository, base_config)

    def set_base_values(self, base_params):
        """Set base parameters for compatibility."""
        self.base_params = base_params
        config_params = {
            k: v for k, v in base_params.items() if k not in ["sentences", "num_sentences"]
        }
        self.runner.base_config = ModelConfig(**config_params)

    def get_model(
        self,
        sentences_tokenized,
        params,
        num_sentences,
        base_dir=None,
        verbose=False,
    ):
        """Legacy method to get a model."""
        config_params = {k: v for k, v in params.items() if k != "sentences"}
        config = ModelConfig(**config_params)
        return self.runner.run_single_experiment(
            sentences=sentences_tokenized,
            config=config,
            num_sentences=num_sentences,
            verbose=verbose,
        )

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
        base_dir=None,
    ):
        """Legacy method to generate a model path."""
        sg = 0 if alg == "CBOW" else 1
        config = ModelConfig(
            vector_size=vector_size,
            window=window,
            epochs=epochs,
            min_count=min_count,
            negative=negative,
            sample=sample,
            hs=hs,
            alpha=alpha,
            seed=seed,
            sg=sg,
        )
        return self.repository.get_model_path(config, num_sentences, sentence_length)

    def create_parameter_sweep_models(
        self,
        num_sentences,
        sweep_values,
        sweep_param,
        base_params,
        base_dir=None,
        verbose=False,
    ):
        """Legacy method for parameter sweep."""
        return self.runner.create_parameter_sweep_models(
            sentences=base_params["sentences"],
            sweep_param=sweep_param,
            sweep_values=sweep_values,
            num_sentences=num_sentences,
            verbose=verbose,
        )

    def create_many_models(
        self,
        models,
        n_sentences_tokenized,
        sentences_tokenized,
        n_used_sentences=10000,
        **kwargs,
    ):
        """Legacy method to create many models."""
        self.base_params["sentences"] = sentences_tokenized
        result_models, categories = self.runner.create_many_models(
            sentences=sentences_tokenized,
            n_sentences_tokenized=n_sentences_tokenized,
            n_used_sentences=n_used_sentences,
            **kwargs,
        )
        models.update(result_models)
        return models, categories
