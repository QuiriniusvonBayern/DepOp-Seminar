import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import umap
import math
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity


def format_title(info, items_per_line=5):
    """Format a title with multiple parameters into multiple lines.

    Args:
        info: Dictionary containing parameter names and values
        items_per_line: Maximum number of items to display per line

    Returns:
        Formatted title string with line breaks
    """
    lines = []
    items = [f"{key}: {value}" for key, value in info.items()]

    for i in range(0, len(items), items_per_line):
        line_items = items[i:i + items_per_line]
        lines.append(", ".join(line_items))

    return "\n".join(lines)


def get_filtered_words(model, filter_prefix, prefix="key_"):
    """Retrieve words from model vocabulary, optionally filtering by prefix.

    Args:
        model: Word2Vec model instance
        filter_prefix: If True, exclude words starting with prefix
        prefix: Prefix string to filter out

    Returns:
        List of words from model vocabulary
    """
    words = list(model.wv.key_to_index.keys())
    if filter_prefix:
        return [word for word in words if not word.startswith(prefix)]
    else:
        return words


def visualize_all_models(models_list, method="tsne", max_cols=2):
    """Visualize multiple word embedding models in a grid layout.

    Uses dimensionality reduction (t-SNE or UMAP) to project word vectors
    into 2D space for visualization.

    Args:
        models_list: List of [model, info_dict] pairs
        method: Dimensionality reduction method ("tsne" or "umap")
        max_cols: Maximum number of columns in the grid
    """
    n_models = len(models_list)

    n_cols = min(max_cols, n_models)
    n_rows = math.ceil(n_models / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6 * n_cols, 5 * n_rows),
        dpi=120,
        constrained_layout=True
    )

    if n_models == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    for idx, (model, info) in enumerate(models_list):
        common_words = get_filtered_words(model, True)
        if idx < len(axes):
            ax = axes[idx]

            available_words = [word for word in common_words if word in model.wv.key_to_index]

            if len(available_words) >= 5:
                vectors = np.array([model.wv[word] for word in available_words])

                if method == "tsne":
                    reducer = TSNE(n_components=2, random_state=42, perplexity=min(5, len(available_words) - 1))
                    reduced = reducer.fit_transform(vectors)
                elif method == "umap":
                    reducer = umap.UMAP()
                    reduced = reducer.fit_transform(vectors)
                else:
                    raise ValueError(f"Unknown method: {method}")

                ax.scatter(reduced[:, 0], reduced[:, 1], alpha=0.7, s=60)

                for i, word in enumerate(available_words):
                    ax.annotate(word, xy=(reduced[i, 0], reduced[i, 1]),
                                fontsize=6, alpha=0.8,
                                xytext=(5, 5), textcoords='offset points')

                title = format_title(info, items_per_line=3)
                ax.set_title(title, fontsize=10)
                ax.grid(True, alpha=0.3)

            else:
                ax.text(0.5, 0.5, f"Model {idx + 1}\nInsufficient words",
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"Model {idx + 1}", fontsize=10)

    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    plt.show()
    print("Visualization completed successfully.")


def visualize_model_comparison(models_list):
    """Display cosine similarity matrices for multiple models.

    Creates heatmaps showing pairwise cosine similarities between
    selected word vectors for each model.

    Args:
        models_list: List of [model, info_dict] pairs
    """
    n_models = len(models_list)

    n_cols = 2
    n_rows = (n_models + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

    if n_rows == 1:
        axes = axes.reshape(1, -1)

    comparison_words = [
        'Balance_Cluster_1', 'Balance_Cluster_5',
        'Age_Young_Adults', 'Age_Middle_aged',
        'Salarie_Low', 'Salarie_High',
        'Credit_Score_Poor', 'Credit_Score_Excellent'
    ]

    for idx, (model, info) in enumerate(models_list):
        row = idx // n_cols
        col = idx % n_cols

        ax = axes[row, col]

        available_words = [word for word in comparison_words if word in model.wv.key_to_index]

        if len(available_words) >= 3:
            vectors = np.array([model.wv[word] for word in available_words])
            similarity_matrix = cosine_similarity(vectors)

            sns.heatmap(similarity_matrix,
                        xticklabels=available_words,
                        yticklabels=available_words,
                        annot=True, fmt='.2f',
                        cmap='coolwarm', center=0,
                        ax=ax, cbar=False)

            title = f"Model {idx + 1}\n"
            keys = list(info.keys())

            for i, key in enumerate(keys):
                if key in info:
                    title += f"{key}: {info[key]}, "
                    if i >= 2:
                        title += f"\n"
            title = title.rstrip(', ')
            ax.set_title(title, fontsize=10)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        else:
            ax.text(0.5, 0.5, "Insufficient words\nfor comparison",
                    ha='center', va='center', transform=ax.transAxes)

            title = f"Model {idx + 1}\n"
            keys = list(info.keys())

            for i, key in enumerate(keys):
                if key in info:
                    title += f"{key}: {info[key]}, "
                    if i >= 2:
                        title += f"\n"
            title = title.rstrip(', ')
            ax.set_title(title, fontsize=10)

    for idx in range(n_models, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)

    plt.show()


def shorten_key(key, max_len=5):
    """Truncate dictionary keys to reduce table width.

    Args:
        key: Original key string
        max_len: Maximum length for truncated key

    Returns:
        Truncated key string
    """
    if len(key) <= max_len:
        return key

    parts = key.split()

    if len(parts) == 1:
        return key[:max_len]

    if len(parts) == 2:
        remaining = max_len - 1
        half = remaining // 2
        part1 = parts[0][:half]
        part2 = parts[1][:remaining - len(part1)]
        return f"{part1} {part2}"

    return key[:max_len]


def print_model_metrics(models_list):
    """Print a formatted table of metrics for all models.

    Dynamically adjusts column widths based on content.

    Args:
        models_list: List of [model, info_dict] pairs
    """
    if not models_list:
        print("No models available.")
        return

    all_keys = set()
    for _, info in models_list:
        all_keys.update(info.keys())
    all_keys = list(all_keys)
    all_keys.append("Vocab Size")

    shortened_keys = {key: shorten_key(key) for key in all_keys}

    table_values = []
    for model, info in models_list:
        row = {}
        for key in all_keys:
            if key == "Vocab Size":
                row[key] = str(len(model.wv.key_to_index))
            else:
                row[key] = str(info.get(key, "N/A"))
        table_values.append(row)

    col_widths = {}
    for key in all_keys:
        max_value_len = max(len(row[key]) for row in table_values)
        col_widths[key] = max(len(shortened_keys[key]), max_value_len)

    header = f"{'Model':<5}  " + "  ".join(
        f"{shortened_keys[key]:<{col_widths[key]}}" for key in all_keys
    )
    print(header)
    print("-" * len(header))

    for i, row in enumerate(table_values, start=1):
        line = f"{i:<5}  " + "  ".join(
            f"{row[key]:<{col_widths[key]}}" for key in all_keys
        )
        print(line)