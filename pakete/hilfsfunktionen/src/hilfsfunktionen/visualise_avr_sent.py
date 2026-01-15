import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import umap
import math
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from typing import Iterable, Tuple, List

def format_title(info, items_per_line=4):
    """Formatiert einen Titel mit mehreren Parametern in mehreren Zeilen."""
    lines = []
    items = [f"{key}: {value}" for key, value in info.items()]
    
    # Aufteilen in Zeilen mit je items_per_line Einträgen
    for i in range(0, len(items), items_per_line):
        line_items = items[i:i + items_per_line]
        lines.append(", ".join(line_items))
    
    return "\n".join(lines)

def extract_words_and_vectors(vector_series, filter, prefix="key_"):
    words = list(vector_series.wv.key_to_index.keys())
    if filter:
        return [w for w in words if not w.startswith(prefix)]
    else:
        return words


def visualize_all_avr_sent(models_list: list, method: str="tsne", max_cols: int=2, nr_of_vectors: int=100):
    """
    Visualisiert alle Modelle in einem automatisch angepassten Grid
    
    Args:
        models_list: Liste von [vector_series, info_dict] Paaren
        max_cols: Maximale Anzahl an Spalten
    """
    n_models = len(models_list)
    
    # Berechne optimale Grid-Größe
    n_cols = min(max_cols, n_models)
    n_rows = math.ceil(n_models / n_cols)
    
    fig, axes = plt.subplots(
        n_rows, 
        n_cols,
        figsize=(6*n_cols, 5*n_rows),
        dpi=120,
        constrained_layout=True
    )
    
    # Falls nur ein Plot, mache axes zu einer Liste
    if n_models == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.flatten()
    elif n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for idx, (vector_df, info) in enumerate(models_list):
        if idx < len(axes):
            ax = axes[idx]
            vector_df = vector_df.iloc[:nr_of_vectors]
            word_series = vector_df.get('key')
            vector_series = vector_df.get('vector')
            vectors = np.vstack(vector_series.values)
            churn_mask = vector_df['churned'].values
            match method:
                case "tsne":
                    tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(vector_series)-1))
                    reduced = tsne.fit_transform(vectors)
                case "umap":
                    reducer = umap.UMAP()
                    reduced = reducer.fit_transform(vectors)
                case _:
                    raise ValueError(f"Unbekannte Methode: {method}")
            
            # Plot
            reduced_churned = reduced[churn_mask]

            ax.scatter(reduced[:, 0], reduced[:, 1], alpha=0.7, s=60, c='green')
            ax.scatter(reduced_churned[:, 0], reduced_churned[:, 1], alpha=0.7, s=60, c='red')
            # Annotationen
            for i, word in enumerate(word_series.values):
                ax.annotate(word, xy=(reduced[i, 0], reduced[i, 1]), 
                            fontsize=6, alpha=0.8, 
                            xytext=(5, 5), textcoords='offset points')
            
            # Titel mit Modell-Informationen
            title = f"vector_series {idx+1}\n"
            
            title = format_title(info, items_per_line=3)
            ax.set_title(title, fontsize=10)
            ax.grid(True, alpha=0.3)
    
    # Verstecke leere Subplots
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)
    
    #plt.tight_layout()
    plt.show()


def visualize_model_comparison(models_list):
    """
    Zeigt Ähnlichkeits-Matrizen für alle Modelle im Vergleich
    Immer zwei Grafiken nebeneinander
    """
    n_models = len(models_list)
    
    # Immer zwei Spalten (Grafiken nebeneinander)
    n_cols = 2
    n_rows = (n_models + 1) // 2  # Aufrunden für ungerade Anzahl
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    
    # Falls nur eine Zeile, axes als 2D-Array behandeln
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (vector_series, info) in enumerate(models_list):
        # Position berechnen
        row = idx // n_cols
        col = idx % n_cols
        
        ax = axes[row, col]
        
        # Verfügbare Wörter
        available_words = [w for w in comparison_words if w in vector_series.wv.key_to_index]
        
        if len(available_words) >= 3:
            # Ähnlichkeitsmatrix berechnen
            vectors = np.array([vector_series.wv[w] for w in available_words])
            similarity_matrix = cosine_similarity(vectors)
            
            # Heatmap plotten
            sns.heatmap(similarity_matrix, 
                       xticklabels=available_words,
                       yticklabels=available_words,
                       annot=True, fmt='.2f',
                       cmap='coolwarm', center=0,
                       ax=ax, cbar=False)
            
            # Titel mit Modell-Informationen
            title = f"vector_series {idx+1}\n"
            keys = list(info.keys())
            
            for i, key in enumerate(keys):
                if key in info:
                    title += f"{key}: {info[key]}, "
                    if i >= 2:  # Nach den ersten 3 Einträgen Umbruch
                        title += f"\n"
            title = title.rstrip(', ')  
            ax.set_title(title, fontsize=10)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        else:
            ax.text(0.5, 0.5, "Nicht genug Wörter\nfür Vergleich", 
                   ha='center', va='center', transform=ax.transAxes)
            
            # Titel auch für Modelle ohne Vergleich
            title = f"vector_series {idx+1}\n"
            keys = list(info.keys())
            
            for i, key in enumerate(keys):
                if key in info:
                    title += f"{key}: {info[key]}, "
                    if i >= 2:
                        title += f"\n"
            title = title.rstrip(', ')
            ax.set_title(title, fontsize=10)
    
    # Leere Subplots ausblenden, falls ungerade Anzahl an Modellen
    for idx in range(n_models, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)
    
    #plt.tight_layout()
    plt.show()


def shorten_key(key, max_len=5):
    """Kürzt Keys automatisch, um Tabellenbreite zu reduzieren."""
    if len(key) <= max_len:
        return key

    parts = key.split()

    if len(parts) == 1:
        return key[:max_len]

    if len(parts) == 2:
        remaining = max_len - 1
        half = remaining // 2
        w1 = parts[0][:half]
        w2 = parts[1][:remaining - len(w1)]
        return f"{w1} {w2}"

    return key[:max_len]


def print_avr_sent_metrics(models_list):
    """
    Passt die Tabellenbreite automatisch an die tatsächlichen Inhalte an.
    Keine Spalte ist breiter als nötig.
    """
    if not models_list:
        print("Keine Modelle vorhanden.")
        return

    # Alle Keys einsammeln
    all_keys = set()
    for _, info in models_list:
        all_keys.update(info.keys())
    all_keys = list(all_keys)
    all_keys.append("Vocab Size")

    # Keys kürzen
    shortened_keys = {key: shorten_key(key) for key in all_keys}

    # Werte vorbereiten
    table_values = []
    for vector_series, info in models_list:
        row = {}
        for key in all_keys:
            if key == "Vocab Size":
                row[key] = str(len(vector_series.index))
            else:
                row[key] = str(info.get(key, "N/A"))
        table_values.append(row)

    # Dynamische Spaltenbreite bestimmen:
    # Breite = max(Key-Länge, maximale Datenlänge in dieser Spalte)
    col_widths = {}
    for key in all_keys:
        max_value_len = max(len(row[key]) for row in table_values)
        col_widths[key] = max(len(shortened_keys[key]), max_value_len)

    # Kopfzeile
    header = f"{'vector_series':<5}  " + "  ".join(
        f"{shortened_keys[key]:<{col_widths[key]}}" for key in all_keys
    )
    print(header)
    print("-" * len(header))

    # Tabellenzeilen
    for i, row in enumerate(table_values, start=1):
        line = f"{i:<5}  " + "  ".join(
            f"{row[key]:<{col_widths[key]}}" for key in all_keys
        )
        print(line)
