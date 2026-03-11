import networkx as nx
import matplotlib.pyplot as plt


def visualize_graph(data):

    edge_index = data.edge_index.numpy()

    G = nx.Graph()

    for i in range(edge_index.shape[1]):
        G.add_edge(edge_index[0, i], edge_index[1, i])

    plt.figure(figsize=(8,6))

    nx.draw(G,
            node_color="lightblue",
            with_labels=True,
            node_size=500)

    plt.show()
