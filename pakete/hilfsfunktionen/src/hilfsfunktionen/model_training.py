import os
from pathlib import Path
from gensim.test.utils import common_texts
from itertools import combinations
from numpy.linalg import norm
from gensim.models import Word2Vec
import time
from datetime import timedelta

# Word2Vec trainieren
def get_model(sentences_tokenized, vector_size, window, Skip_Gram, min_count, workers, epochs, num_sentences):
    sentences_selection = sentences_tokenized[:num_sentences]
    sentence_length = len(sentences_selection[0])
    alg = "CBOW"
    if(Skip_Gram == 1):
        alg = "SGNS"
    name = get_model_path(alg, vector_size, window, epochs, num_sentences, sentence_length)
    print("Started Model-Creation with name: "+ str(name))
    if not (os.path.exists(name)):
        startzeit = time.time()
        print("Model has to be created new.")
        print("Startzeit: "+ str(time.ctime(startzeit)))
        
        model = Word2Vec(
            sentences=sentences_selection,
            vector_size=vector_size,    
            window=window,              
            sg=Skip_Gram,               
            min_count=min_count,
            workers=workers,
            epochs=epochs
        )
        model.save(name)

        endzeit = time.time()
        verstrichene_zeit = endzeit - startzeit
        print("Fertig um: "+str(time.ctime(endzeit)))
        print(f"Nach: {verstrichene_zeit:.4f} Sekunden")
        print(f"created new Model with the name: {name}")
    else:
        model = Word2Vec.load(name)
        print(f"loaded model with name: {name}")
    return [model, { "Vector Size" : vector_size, "Window":window, "Algorithmus": alg, "Kleinste Worthäufigkeit": min_count, "Workers":workers, "Epochs":epochs, "Satzanzahl":num_sentences, "Satzlänge":sentence_length}]
    

def get_model_path(alg, vector_size, window, epochs, num_sentences, sentence_length, base_dir=None):
    """
    base_dir: Optional. Wenn None, wird Projekt-Root automatisch erkannt
    """
    from pathlib import Path
    
    filename = f"{alg}_vs_{vector_size}_win_{window}_ep_{epochs}_sentences_{num_sentences}_sentence_length_{sentence_length}.model"
    
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