from pyvis.network import Network
import networkx as nx
from IPython.display import display, HTML


def visualize_interactive_graph(data, image_paths, edge_mask=None):

    G = nx.Graph()

    edge_index = data.edge_index.cpu().numpy()

    for i in range(data.num_nodes):

        img_path = image_paths[i]

        G.add_node(
            i,
            title=f'<img src="{img_path}" width="120">',
            label=str(i)
        )

    for i in range(edge_index.shape[1]):

        src = int(edge_index[0][i])
        dst = int(edge_index[1][i])

        color = "gray"
        width = 1

        if edge_mask is not None and i < len(edge_mask):

            importance = edge_mask[i].item()

            if importance > 0.5:
                color = "red"
                width = 4

        G.add_edge(src, dst, color=color, width=width)

    net = Network(
        height="750px",
        width="100%",
        notebook=True,
        cdn_resources="in_line"
    )

    net.from_nx(G)

    net.show("graph.html")

    display(HTML("graph.html"))
