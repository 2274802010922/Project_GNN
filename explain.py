import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.explain import Explainer, GNNExplainer


class PrototypeClassifierWrapper(torch.nn.Module):
    def __init__(self, encoder, prototypes, temperature=0.1):
        super().__init__()
        self.encoder = encoder
        self.register_buffer("prototypes", F.normalize(prototypes, p=2, dim=1))
        self.temperature = temperature

    def forward(self, x, edge_index, edge_weight=None):
        embeddings = self.encoder(x, edge_index, edge_weight=edge_weight)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return torch.mm(embeddings, self.prototypes.t()) / self.temperature


def compute_full_graph_prototypes(encoder, data):
    device = next(encoder.parameters()).device
    data = data.to(device)
    encoder.eval()

    with torch.no_grad():
        embeddings = encoder(data.x, data.edge_index, data.edge_weight)
        prototypes = []
        for class_id in range(len(data.class_names)):
            mask = data.y == class_id
            if int(mask.sum()) > 0:
                prototypes.append(embeddings[mask].mean(dim=0))
            else:
                prototypes.append(torch.zeros(embeddings.shape[1], device=device))
        prototypes = torch.stack(prototypes, dim=0)

    return prototypes


def run_gnn_explainer(encoder, data, node_idx=None, output_dir="outputs", temperature=0.1):
    os.makedirs(output_dir, exist_ok=True)

    device = next(encoder.parameters()).device
    data = data.to(device)

    if node_idx is None:
        degrees = torch.bincount(data.edge_index[0], minlength=data.num_nodes)
        node_idx = int(torch.argmax(degrees).item())

    prototypes = compute_full_graph_prototypes(encoder, data)
    wrapper = PrototypeClassifierWrapper(encoder, prototypes, temperature=temperature).to(device)
    wrapper.eval()

    with torch.no_grad():
        logits = wrapper(data.x, data.edge_index, data.edge_weight)
        pred_class = int(logits[node_idx].argmax().item())
        true_class = int(data.y[node_idx].item())

    print(f"Running GNNExplainer for node {node_idx}")
    print(f"True class: {data.class_names[true_class]}")
    print(f"Predicted class: {data.class_names[pred_class]}")

    explainer = Explainer(
        model=wrapper,
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

    graph = nx.Graph()
    for i in range(data.num_nodes):
        graph.add_node(i)

    for i in range(edge_index.shape[1]):
        src = int(edge_index[0, i])
        dst = int(edge_index[1, i])
        score = float(edge_mask[i])
        if score >= threshold:
            graph.add_edge(src, dst, weight=score)

    if node_idx not in graph:
        graph.add_node(node_idx)

    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(graph, seed=42)
    node_colors = ["orange" if n == node_idx else "lightblue" for n in graph.nodes()]
    edge_widths = [2.0 + 3.0 * graph[u][v]["weight"] for u, v in graph.edges()] if graph.number_of_edges() > 0 else None

    nx.draw(
        graph,
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
