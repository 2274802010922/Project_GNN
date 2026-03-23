import os

import torch

from dataset import download_dataset, load_image_paths_and_labels
from explain import run_gnn_explainer
from feature_extractor import extract_features
from gnn_model import GNN
from graph_builder import build_graph
from interactive_graph import visualize_interactive_graph
from train import train_model
from visualize import (
    plot_class_distribution,
    plot_degree_distribution,
    plot_similarity_distribution,
    plot_tsne_embeddings,
    visualize_graph,
)


def main():
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    print("\n===== STEP 1: DOWNLOAD DATASET =====")
    dataset_path = download_dataset()

    print("\n===== STEP 2: LOAD IMAGE PATHS + LABELS =====")
    image_paths, labels, class_names = load_image_paths_and_labels(dataset_path)

    print("\n===== STEP 3: EXTRACT FEATURES =====")
    features = extract_features(image_paths, batch_size=32)

    print("\n===== STEP 4: BUILD GRAPH =====")
    graph = build_graph(
        features=features,
        labels=labels,
        image_paths=image_paths,
        class_names=class_names,
        k=5,
    )

    print("\n===== STEP 5: DATASET / GRAPH VISUALIZATION =====")
    plot_class_distribution(labels, class_names, output_dir=output_dir)
    plot_degree_distribution(graph, output_dir=output_dir)
    plot_similarity_distribution(graph, output_dir=output_dir)
    visualize_graph(graph, image_paths, max_nodes=36, output_dir=output_dir)

    print("\n===== STEP 6: TRAIN GNN =====")
    model = GNN(
        input_dim=graph.x.shape[1],
        num_classes=len(class_names),
        hidden_dim=128,
        dropout=0.3,
    )
    model, history, logits, test_predictions = train_model(
        model,
        graph,
        epochs=80,
        lr=1e-3,
        weight_decay=1e-4,
        output_dir=output_dir,
    )

    print("\n===== STEP 7: t-SNE BEFORE / AFTER GNN =====")
    model.eval()
    device = next(model.parameters()).device
    graph_device = graph.to(device)
    with torch.no_grad():
        embeddings_after = model.get_embeddings(
            graph_device.x,
            graph_device.edge_index,
            graph_device.edge_weight,
        ).cpu().numpy()

    plot_tsne_embeddings(
        features_before=features,
        features_after=embeddings_after,
        labels=labels,
        class_names=class_names,
        output_dir=output_dir,
    )

    print("\n===== STEP 8: RUN GNN EXPLAINER =====")
    try:
        node_mask, edge_mask, explained_node = run_gnn_explainer(
            model,
            graph,
            node_idx=None,
            output_dir=output_dir,
        )
    except Exception as e:
        print(f"GNNExplainer failed: {e}")
        edge_mask = None

    print("\n===== STEP 9: INTERACTIVE GRAPH =====")
    try:
        visualize_interactive_graph(graph, image_paths, edge_mask=edge_mask, output_dir=output_dir)
    except Exception as e:
        print(f"Interactive graph failed: {e}")

    print("\nPipeline completed successfully.")
    print(f"All figures are saved in: {output_dir}")


if __name__ == "__main__":
    main()
