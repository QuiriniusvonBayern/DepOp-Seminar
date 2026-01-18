import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import time
from gensim.models import Word2Vec
from itertools import product


@dataclass(frozen=True)
class ModelConfig:
    """Immutable configuration for Word2Vec models"""
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
        """converts Config zu Dictionary für Word2Vec"""
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
            "sg": self.sg
        }
    
    def get_algorithm_name(self) -> str:
        """Gibt den Algorithmusnamen zurück"""
        return "CBOW" if self.sg == 0 else "SGNS"


class ModelTrainer:
    """Trainiert genau ein Word2Vec Modell"""
    
    def train_model(
        self,
        sentences: List[List[str]],
        config: ModelConfig,
        verbose: bool = False
    ) -> Word2Vec:
        """
        Trainiert ein Word2Vec Modell mit der gegebenen Konfiguration.
        
        Args:
            sentences: Liste von tokenisierten Sätzen
            config: ModelConfig mit allen Hyperparametern
            verbose: Ob Trainingsfortschritt ausgegeben werden soll
            
        Returns:
            Trainiertes Word2Vec Modell
        """
        if not sentences:
            raise ValueError("Keine Sätze zum Trainieren übergeben")
        
        if verbose:
            print(f"Trainiere Modell mit {len(sentences)} Sätzen...")
            startzeit = time.time()
            print("Startzeit: " + str(time.ctime(startzeit)))
        
        model = Word2Vec(
            sentences=sentences,
            **config.to_dict()
        )
        
        if verbose:
            endzeit = time.time()
            verstrichene_zeit = endzeit - startzeit
            minuten, sekunden = divmod(verstrichene_zeit, 60)
            print("Fertig um: " + str(time.ctime(endzeit)))
            print(f"Nach: {int(minuten)} Minuten und {int(sekunden)} Sekunden")
        
        return model


class ModelRepository:
    """Verwaltet Persistenz und Wiederauffindbarkeit von Modellen"""
    
    def __init__(self, base_dir: Optional[Path] = None):
        """
        Args:
            base_dir: Basis-Verzeichnis für Modelle. 
                     Wenn None, wird automatisch gesucht.
        """
        self.base_dir = self._find_base_dir(base_dir)
    
    def _find_base_dir(self, base_dir: Optional[Path]) -> Path:
        """Findet oder erstellt das Basis-Verzeichnis für Modelle"""
        if base_dir is None:
            base_dir = Path.cwd()
            
            # Suche nach 'models' Ordner in Eltern-Verzeichnissen
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
        sentence_length: int
    ) -> Path:
        """
        Erstellt eindeutigen Modell-Dateinamen basierend auf allen Parametern.
        
        Args:
            config: ModelConfig
            num_sentences: Anzahl trainierter Sätze
            sentence_length: Länge der Sätze
            
        Returns:
            Pfad zur Modell-Datei
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
        verbose: bool = False
    ) -> Path:
        """
        Speichert ein Modell.
        
        Args:
            model: Zu speicherndes Word2Vec Modell
            config: ModelConfig des Modells
            num_sentences: Anzahl trainierter Sätze
            sentence_length: Länge der Sätze
            verbose: Ob Status ausgegeben werden soll
            
        Returns:
            Pfad zur gespeicherten Datei
        """
        path = self.get_model_path(config, num_sentences, sentence_length)
        model.save(str(path))
        
        if verbose:
            print(f"Modell gespeichert: {path}")
        
        return path
    
    def load(
        self,
        config: ModelConfig,
        num_sentences: int,
        sentence_length: int,
        verbose: bool = False
    ) -> Optional[Word2Vec]:
        """
        Lädt ein Modell, falls es existiert.
        
        Args:
            config: ModelConfig des gewünschten Modells
            num_sentences: Anzahl trainierter Sätze
            sentence_length: Länge der Sätze
            verbose: Ob Status ausgegeben werden soll
            
        Returns:
            Word2Vec Modell oder None, falls nicht gefunden
        """
        path = self.get_model_path(config, num_sentences, sentence_length)
        
        if not path.exists():
            return None
        
        model = Word2Vec.load(str(path))
        
        if verbose:
            print(f"Modell geladen: {path}")
        
        return model
    
    def exists(
        self,
        config: ModelConfig,
        num_sentences: int,
        sentence_length: int
    ) -> bool:
        """Prüft, ob ein Modell bereits existiert"""
        path = self.get_model_path(config, num_sentences, sentence_length)
        return path.exists()


class ExperimentRunner:
    """Orchestriert viele Trainingsläufe für Parameter-Sweeps"""
    
    def __init__(
        self,
        trainer: ModelTrainer,
        repository: ModelRepository,
        base_config: ModelConfig
    ):
        """
        Args:
            trainer: ModelTrainer Instanz
            repository: ModelRepository Instanz
            base_config: Basis-Konfiguration für Experimente
        """
        self.trainer = trainer
        self.repository = repository
        self.base_config = base_config
    
    def run_single_experiment(
        self,
        sentences: List[List[str]],
        config: ModelConfig,
        num_sentences: int,
        verbose: bool = False
    ) -> Tuple[Word2Vec, Dict[str, Any]]:
        """
        Führt ein einzelnes Experiment durch (mit Caching).
        
        Args:
            sentences: Tokenisierte Sätze
            config: ModelConfig für dieses Experiment
            num_sentences: Anzahl zu verwendender Sätze
            verbose: Ob Status ausgegeben werden soll
            
        Returns:
            Tuple aus (trainiertes Modell, Modell-Metadaten)
        """
        sentences_selection = sentences[:num_sentences]
        sentence_length = len(sentences_selection[0]) if sentences_selection else 0
        
        if verbose:
            print(f"Started Model-Creation with: {config.get_algorithm_name()}, "
                  f"vs={config.vector_size}, win={config.window}, ep={config.epochs}")
        
        # Versuche Modell zu laden
        model = self.repository.load(config, num_sentences, sentence_length, verbose)
        
        if model is None:
            # Modell muss neu trainiert werden
            if verbose:
                print("Model has to be created new.")
            
            model = self.trainer.train_model(sentences_selection, config, verbose)
            self.repository.save(model, config, num_sentences, sentence_length, verbose)
            
            if verbose:
                print("Created new model.")
        else:
            if verbose:
                print("Loaded model.")
        
        # Erstelle Metadaten
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
            "Seed": config.seed
        }
        
        return model, metadata
    
    def create_parameter_sweep_models(
        self,
        sentences: List[List[str]],
        sweep_param: str,
        sweep_values: List[Any],
        num_sentences: int,
        verbose: bool = False
    ) -> List[Tuple[Word2Vec, Dict[str, Any]]]:
        """
        Erstellt Modelle für einen Parameter-Sweep.
        
        Args:
            sentences: Tokenisierte Sätze
            sweep_param: Name des zu variierenden Parameters
            sweep_values: Liste von Werten für den Parameter
            num_sentences: Anzahl zu verwendender Sätze
            verbose: Ob Status ausgegeben werden soll
            
        Returns:
            Liste von (Modell, Metadaten) Tupeln
        """
        if verbose:
            print(f"...Creating {sweep_param} models...")
        
        models = []
        
        for value in sweep_values:
            for alg in [0, 1]:  # CBOW und Skip-gram
                # Erstelle neue Config mit variiertem Parameter
                config_dict = self.base_config.to_dict()
                config_dict[sweep_param] = value
                config_dict["sg"] = alg
                
                config = ModelConfig(**config_dict)
                
                model, metadata = self.run_single_experiment(
                    sentences=sentences,
                    config=config,
                    num_sentences=num_sentences,
                    verbose=verbose
                )
                
                models.append((model, metadata))
        
        if verbose:
            print(f"...Finished Creating {sweep_param} models...")
        
        return models
    
    def create_parameter_sweep_models(
            self,
            sentences: List[List[str]],
            sweep_param: str,
            sweep_values: List[Any],
            num_sentences: int,
            verbose: bool = False
        ) -> List[Tuple[Word2Vec, Dict[str, Any]]]:
            """
            Erstellt Modelle für einen Parameter-Sweep.
            
            Args:
                sentences: Tokenisierte Sätze
                sweep_param: Name des zu variierenden Parameters
                sweep_values: Liste von Werten für den Parameter
                num_sentences: Anzahl zu verwendender Sätze
                verbose: Ob Status ausgegeben werden soll
                
            Returns:
                Liste von (Modell, Metadaten) Tupeln
            """
            if verbose:
                print(f"...Creating {sweep_param} models...")
            
            models = []
            
            for value in sweep_values:
                for alg in [0, 1]:  # CBOW und Skip-gram
                    # Erstelle neue Config mit variiertem Parameter
                    config_dict = self.base_config.to_dict()
                    config_dict[sweep_param] = value
                    config_dict["sg"] = alg
                    
                    config = ModelConfig(**config_dict)
                    
                    model, metadata = self.run_single_experiment(
                        sentences=sentences,
                        config=config,
                        num_sentences=num_sentences,
                        verbose=verbose
                    )
                    
                    models.append((model, metadata))
            
            if verbose:
                print(f"...Finished Creating {sweep_param} models...")
            
            return models


    def run_grid_search(
        self,
        sentences: List[List[str]],
        param_grid: Dict[str, List[Any]],
        num_sentences: int,
        verbose: bool = False
    ):
        results = []

        grid_keys = list(param_grid.keys())
        grid_values = list(param_grid.values())

        for values in product(*grid_values):
            grid_params = dict(zip(grid_keys, values))

            # Base-Config als Ausgangspunkt
            config_dict = self.base_config.to_dict()
            config_dict.update(grid_params)

            config = ModelConfig(**config_dict)

            model, metadata = self.run_single_experiment(
                sentences=sentences,
                config=config,
                num_sentences=num_sentences,
                verbose=verbose
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
        verbose: bool = False
    ) -> Tuple[Dict[str, List], List[str]]:
        """
        Erstellt viele Modelle mit verschiedenen Parameter-Kombinationen.
        
        Args:
            sentences: Tokenisierte Sätze
            n_sentences_tokenized: Dictionary mit verschieden langen tokenisierten Sätzen
            n_used_sentences: Standard-Anzahl zu verwendender Sätze
            vector_sizes: Liste von Vector Sizes zum Testen
            window_sizes: Liste von Window Sizes zum Testen
            num_epochs: Liste von Epoch-Zahlen zum Testen
            negatives: Liste von Negative Sampling Werten
            samples: Liste von Sample-Werten
            hs_values: Liste von Hierarchical Softmax Werten
            alphas: Liste von Learning Rates
            seeds: Liste von Random Seeds
            verbose: Ob Status ausgegeben werden soll
            
        Returns:
            Tuple aus (models dictionary, categories list)
        """
        # Default-Werte setzen
        if vector_sizes is None:
            vector_sizes = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 
                          85, 90, 95, 100, 110, 120, 130, 140, 150, 180, 225, 
                          250, 275, 300]
        if window_sizes is None:
            window_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                          16, 17, 18, 20]
        if num_epochs is None:
            num_epochs = [3, 5, 7, 10, 13, 15, 17, 20, 23, 25, 28, 30, 33, 35,
                        38, 40, 43, 45, 48, 50, 55, 60, 65, 70, 75, 80, 90,
                        100, 110, 120, 130, 140, 150, 175, 200, 250]
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
        
        # Definiere Experimente
        experiments = {
            "sentence": ("sentences", list(n_sentences_tokenized.values())),
            "vector": ("vector_size", vector_sizes),
            "windowsize": ("window", window_sizes),
            "epoch": ("epochs", num_epochs),
            "negative": ("negative", negatives),
            "sample": ("sample", samples),
            "hs": ("hs", hs_values),
            "alpha": ("alpha", alphas),
            #"seed": ("seed", seeds),
        }
        
        models = {}
        
        print("Model Creation started.")
        
        for name, (param, values) in experiments.items():
            if verbose:
                print(f"...Creating {name} models...")
            
            # Für sentence-Sweep: verwende verschiedene Satzanzahlen
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
                            verbose=verbose
                        )
                        sweep_models.append((model, metadata))
                models[name] = sweep_models
            else:
                # Für andere Parameter: normaler Sweep
                models[name] = self.create_parameter_sweep_models(
                    sentences=sentences,
                    sweep_param=param,
                    sweep_values=values,
                    num_sentences=n_used_sentences,
                    verbose=verbose
                )
            
            if verbose:
                print(f"...Finished Creating {name} models...")
        
        categories = list(models.keys())
        print("Model Creation finished.")
        
        return models, categories


class ModelGenerator:
    """
    Kompatibilitäts-Wrapper für alten Code.
    Delegiert an die neuen Klassen.
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
            "sentences": []
        }
        
        # Initialisiere neue Komponenten
        base_config = ModelConfig(**{k: v for k, v in self.base_params.items() 
                                     if k != "sentences" and k != "num_sentences"})
        self.trainer = ModelTrainer()
        self.repository = ModelRepository(base_dir)
        self.runner = ExperimentRunner(self.trainer, self.repository, base_config)
    
    def set_base_values(self, base_params):
        """Setzt Basis-Parameter (für Kompatibilität)"""
        self.base_params = base_params
        config_params = {k: v for k, v in base_params.items() 
                        if k != "sentences" and k != "num_sentences"}
        self.runner.base_config = ModelConfig(**config_params)
    
    def get_model(
        self,
        sentences_tokenized,
        params,
        num_sentences,
        base_dir=None,
        verbose=False
    ):
        """Alte get_model Methode (für Kompatibilität)"""
        config_params = {k: v for k, v in params.items() if k != "sentences"}
        config = ModelConfig(**config_params)
        
        return self.runner.run_single_experiment(
            sentences=sentences_tokenized,
            config=config,
            num_sentences=num_sentences,
            verbose=verbose
        )
    
    def get_model_path(self, alg, vector_size, window, epochs, num_sentences,
                      sentence_length, min_count, negative, sample, hs, 
                      alpha, seed, base_dir=None):
        """Alte get_model_path Methode (für Kompatibilität)"""
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
            sg=sg
        )
        return self.repository.get_model_path(config, num_sentences, sentence_length)
    
    def create_parameter_sweep_models(self, num_sentences, sweep_values, 
                                     sweep_param, base_params, base_dir=None, 
                                     verbose=False):
        """Alte create_parameter_sweep_models Methode (für Kompatibilität)"""
        return self.runner.create_parameter_sweep_models(
            sentences=base_params["sentences"],
            sweep_param=sweep_param,
            sweep_values=sweep_values,
            num_sentences=num_sentences,
            verbose=verbose
        )
    
    def create_many_models(self, models, n_sentences_tokenized, 
                          sentences_tokenized, n_used_sentences=10000,
                          **kwargs):
        """Alte create_many_models Methode (für Kompatibilität)"""
        self.base_params["sentences"] = sentences_tokenized
        
        result_models, categories = self.runner.create_many_models(
            sentences=sentences_tokenized,
            n_sentences_tokenized=n_sentences_tokenized,
            n_used_sentences=n_used_sentences,
            **kwargs
        )
        
        # Aktualisiere models dictionary
        models.update(result_models)
        
        return models, categories