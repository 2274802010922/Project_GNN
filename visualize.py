import matplotlib.pyplot as plt
import networkx as nx
from PIL import Image
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def visualize_graph(data, image_paths):

    G = nx.Graph()

    edge_index = data.edge_index.cpu().numpy()

    # add nodes
    for i in range(len(image_paths)):
        G.add_node(i)

    # add edges
    for i in range(edge_index.shape[1]):
        src = edge_index[0][i]
        dst = edge_index[1][i]
        G.add_edge(src, dst)

    pos = nx.spring_layout(G, seed=42)

    fig, ax = plt.subplots(figsize=(10,8))

    # draw edges only
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3)

    # hide default nodes
    nx.draw_networkx_nodes(G, pos, node_size=0)

    # add images
    for node in G.nodes:

        img = Image.open(image_paths[node])
        img = img.resize((40,40))

        imagebox = OffsetImage(np.array(img), zoom=1)

        ab = AnnotationBbox(
            imagebox,
            pos[node],
            frameon=False
        )

        ax.add_artist(ab)

    ax.set_xlim(min(x for x,y in pos.values())-0.1, max(x for x,y in pos.values())+0.1)
    ax.set_ylim(min(y for x,y in pos.values())-0.1, max(y for x,y in pos.values())+0.1)

    plt.title("Graph Visualization with Images")
    plt.axis("off")
    plt.show()
