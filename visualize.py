import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def visualize_graph(data,explanation):

    G = nx.Graph()

    num_nodes = data.x.shape[0]

    for i in range(num_nodes):
        G.add_node(i)

    edge_index=data.edge_index

    for i in range(edge_index.shape[1]):

        u=edge_index[0,i].item()
        v=edge_index[1,i].item()

        G.add_edge(u,v)

    node_importance = explanation.node_mask.mean(dim=1).detach().cpu().numpy()
    node_importance = node_importance/node_importance.max()

    edge_importance = explanation.edge_mask.detach().cpu().numpy()
    edge_importance = edge_importance/edge_importance.max()

    pos = nx.spring_layout(G,seed=42)

    plt.figure(figsize=(8,6))

    nx.draw_networkx_nodes(
        G,pos,
        node_size=3000*node_importance+200
    )

    threshold=np.percentile(edge_importance,70)

    important_edges=[
        (u,v)
        for (u,v),w in zip(G.edges(),edge_importance)
        if w>=threshold
    ]

    nx.draw_networkx_edges(
        G,pos,
        edgelist=important_edges,
        width=3
    )

    nx.draw_networkx_labels(G,pos)

    plt.title("GNNExplainer Important Subgraph")

    plt.axis("off")

    plt.show()
