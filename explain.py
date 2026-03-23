import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from torch_geometric.explain import Explainer, GNNExplainer


def run_gnn_explainer(model, data, node_idx=None, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)

    device = next(model.parameters()).device
    data = data.to(device)
    model.eval()

    if node_idx is None:
        candidate_indices = torch.where(data.test_mask)[0]
        if len(candidate_indices) == 0:
            candidate_indices = torch.where(data.val_mask)[0]
        if len(candidate_indices) == 0:
            candidate_indices = torch.where(data.train_mask)[0]
        node_idx = int(candidate_indices[0].item())

    with torch.no_grad():
        logits = model(data.x, data.edge_index, data.edge_weight)
        pred_class = int(logits[node_idx].argmax().item())
        true_class = int(data.y[node_idx].item())

    print(f"Running GNNExplainer for node {node_idx}")
    print(f"True class: {data.class_names[true_class]}")
    print(f"Predicted class: {data.class_names[pred_class]}")

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=100),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(
            mode="multiclass_classification",
            task_level="node",
            return_type="raw",
        ),
    )

    explanation = explainer(
        x=data.x,
        edge_index=data.edge_index,
        edge_weight=data.edge_weight,
        index=node_idx,
    )

    node_mask = explanation.node_mask.detach().cpu()
    edge_mask = explanation.edge_mask.detach().cpu()

    plot_edge_importance(edge_mask, output_dir=output_dir)
    visualize_explanation(data.cpu(), edge_mask, node_idx=node_idx, output_dir=output_dir)

    return node_mask, edge_mask, node_idx


def plot_edge_importance(edge_mask, top_k=15, output_dir="outputs"):
    edge_scores = edge_mask.numpy()
    order = np.argsort(edge_scores)[::-1][:top_k]
    top_scores = edge_scores[order]

    plt.figure(figsize=(10, 4))
    plt.bar(range(len(top_scores)), top_scores)
    plt.xlabel("Top edges")
    plt.ylabel("Importance")
    plt.title("Top Edge Importance Scores")
    save_path = os.path.join(output_dir, "explainer_edge_importance.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.show()


def visualize_explanation(data, edge_mask, node_idx, output_dir="outputs", threshold=0.5):
    edge_index = data.edge_index.cpu().numpy()
    edge_mask = edge_mask.cpu().numpy()

    G = nx.Graph()
    for i in range(data.num_nodes):
        G.add_node(i)

    for i in range(edge_index.shape[1]):
        src = int(edge_index[0, i])
        dst = int(edge_index[1, i])
        score = float(edge_mask[i])
        if score >= threshold:
            G.add_edge(src, dst, weight=score)

    if node_idx not in G:
        G.add_node(node_idx)

    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)
    node_colors = ["orange" if n == node_idx else "lightblue" for n in G.nodes()]
    edge_widths = [2.0 + 3.0 * G[u][v]["weight"] for u, v in G.edges()] if G.number_of_edges() > 0 else None

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color=node_colors,
        node_size=500,
        width=edge_widths,
        font_size=8,
    )
    plt.title(f"Explanation Subgraph for Node {node_idx}")
    save_path = os.path.join(output_dir, "explanation_subgraph.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.show()
