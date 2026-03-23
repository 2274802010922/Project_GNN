import os
from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
from sklearn.manifold import TSNE


def plot_class_distribution(labels, class_names, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)

    counts = Counter(labels)
    x = np.arange(len(class_names))
    y = [counts.get(i, 0) for i in x]

    plt.figure(figsize=(12, 4))
    plt.bar(x, y)
    plt.xticks(x, class_names, rotation=90)
    plt.xlabel("Class")
    plt.ylabel("Number of images")
    plt.title("Class Distribution")
    plt.tight_layout()

    save_path = os.path.join(output_dir, "class_distribution.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.show()


def plot_degree_distribution(data, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)

    edge_index = data.edge_index.cpu().numpy()
    degrees = np.zeros(data.num_nodes, dtype=int)
    for src, dst in edge_index.T:
        degrees[src] += 1

    plt.figure(figsize=(8, 4))
    plt.hist(degrees, bins=min(20, max(5, len(np.unique(degrees)))))
    plt.xlabel("Node degree")
    plt.ylabel("Frequency")
    plt.title("Degree Distribution")
    plt.tight_layout()

    save_path = os.path.join(output_dir, "degree_distribution.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.show()


def plot_similarity_distribution(data, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)

    edge_weight = data.edge_weight.cpu().numpy()
    plt.figure(figsize=(8, 4))
    plt.hist(edge_weight, bins=20)
    plt.xlabel("Cosine similarity")
    plt.ylabel("Frequency")
    plt.title("Edge Similarity Distribution")
    plt.tight_layout()

    save_path = os.path.join(output_dir, "similarity_distribution.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.show()


def visualize_graph(data, image_paths, max_nodes=36, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)

    G = nx.Graph()
    edge_index = data.edge_index.cpu().numpy()

    for i in range(data.num_nodes):
        G.add_node(i)

    for i in range(edge_index.shape[1]):
        src = int(edge_index[0, i])
        dst = int(edge_index[1, i])
        G.add_edge(src, dst)

    if data.num_nodes > max_nodes:
        degrees = dict(G.degree())
        selected_nodes = sorted(degrees, key=degrees.get, reverse=True)[:max_nodes]
        G = G.subgraph(selected_nodes).copy()

    pos = nx.spring_layout(G, seed=42)
    fig, ax = plt.subplots(figsize=(12, 10))

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3)
    nx.draw_networkx_nodes(G, pos, node_size=0)

    for node in G.nodes():
        img = Image.open(image_paths[node]).convert("RGB").resize((40, 40))
        imagebox = OffsetImage(np.array(img), zoom=1.0)
        ab = AnnotationBbox(imagebox, pos[node], frameon=False)
        ax.add_artist(ab)

    ax.set_title("Graph Visualization with Images")
    ax.axis("off")

    save_path = os.path.join(output_dir, "graph_with_images.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.show()


def plot_tsne_embeddings(features_before, features_after, labels, class_names, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)

    labels = np.asarray(labels)
    perplexity = max(5, min(30, len(labels) - 1))

    tsne_before = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    emb_before = tsne_before.fit_transform(np.asarray(features_before))

    tsne_after = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    emb_after = tsne_after.fit_transform(np.asarray(features_after))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for class_idx, class_name in enumerate(class_names):
        mask = labels == class_idx
        plt.scatter(emb_before[mask, 0], emb_before[mask, 1], s=18, label=class_name)
    plt.title("t-SNE Before GNN")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")

    plt.subplot(1, 2, 2)
    for class_idx, class_name in enumerate(class_names):
        mask = labels == class_idx
        plt.scatter(emb_after[mask, 0], emb_after[mask, 1], s=18, label=class_name)
    plt.title("t-SNE After GNN")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")

    if len(class_names) <= 12:
        plt.legend(loc="best", fontsize=8)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "tsne_embeddings.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.show()
