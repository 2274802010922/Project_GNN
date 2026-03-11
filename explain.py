import torch
from torch_geometric.explain import Explainer, GNNExplainer
import matplotlib.pyplot as plt
import networkx as nx


def run_gnn_explainer(model, data):

    print("===== STEP 5: RUN GNN EXPLAINER =====")

    model.eval()

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object'
    )

    explanation = explainer(
        x=data.x,
        edge_index=data.edge_index
    )

    node_mask = explanation.node_mask
    edge_mask = explanation.edge_mask

    print("Node importance shape:", node_mask.shape)
    print("Edge importance shape:", edge_mask.shape)

    visualize_explanation(data, edge_mask)

    print("GNNExplainer finished")

    return explanation


def visualize_explanation(data, edge_mask):

    edge_index = data.edge_index.cpu().numpy()

    G = nx.Graph()

    for i in range(data.num_nodes):
        G.add_node(i)

    for i in range(edge_index.shape[1]):
        src = edge_index[0][i]
        dst = edge_index[1][i]

        importance = edge_mask[i].item()

        if importance > 0.5:
            G.add_edge(src, dst, color="red", weight=2)
        else:
            G.add_edge(src, dst, color="gray", weight=0.5)

    pos = nx.spring_layout(G)

    edges = G.edges()
    colors = [G[u][v]['color'] for u,v in edges]
    weights = [G[u][v]['weight'] for u,v in edges]

    plt.figure(figsize=(8,6))

    nx.draw(
        G,
        pos,
        edge_color=colors,
        width=weights,
        node_color="lightblue",
        node_size=500,
        with_labels=True
    )

    plt.title("GNN Explanation Graph")
    plt.show()
