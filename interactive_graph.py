import os

import networkx as nx
from IPython.display import HTML, display
from pyvis.network import Network


def visualize_interactive_graph(data, image_paths, edge_mask=None, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)

    graph = nx.Graph()
    edge_index = data.edge_index.cpu().numpy()

    for i in range(data.num_nodes):
        class_name = data.class_names[int(data.y[i].item())]
        graph.add_node(
            i,
            title=f"Node: {i}<br>Class: {class_name}<br>Image: {os.path.basename(image_paths[i])}",
            label=str(i),
        )

    for i in range(edge_index.shape[1]):
        src = int(edge_index[0, i])
        dst = int(edge_index[1, i])
        width = 1
        color = "gray"

        if edge_mask is not None and i < len(edge_mask):
            importance = float(edge_mask[i].item())
            if importance >= 0.5:
                color = "red"
                width = 2 + 4 * importance

        graph.add_edge(src, dst, color=color, width=width)

    net = Network(height="750px", width="100%", notebook=True, cdn_resources="in_line")
    net.from_nx(graph)

    html_path = os.path.join(output_dir, "interactive_graph.html")
    net.save_graph(html_path)
    print(f"Saved: {html_path}")

    try:
        display(HTML(filename=html_path))
    except Exception:
        pass
