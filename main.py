import os

import torch

from dataset import download_dataset, inspect_dataset_structure, load_image_paths_and_labels
from explain import run_gnn_explainer
from feature_extractor import extract_features
from gnn_model import FewShotGNN
from graph_builder import build_full_graph
from interactive_graph import visualize_interactive_graph
from train import prepare_fewshot_splits, train_fewshot_model
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

    # Few-shot configuration
    k_shot = 1
    q_query = 1
    n_way = 5
    full_graph_k = 5
    episode_graph_k = 3
    epochs = 40
    episodes_per_epoch = 30
    val_episodes = 20
    test_episodes = 50
    seed = 42

    print("\n===== STEP 1: DOWNLOAD DATASET =====")
    dataset_path = download_dataset()

    print("\n===== STEP 2: LOAD IMAGE PATHS + LABELS =====")
    try:
        image_paths, labels, class_names, label_source = load_image_paths_and_labels(dataset_path)
    except Exception as e:
        print("\nLabel loading failed with a helpful diagnostic.")
        print(str(e))
        print("\nInspecting dataset structure so you can see where labels may be stored:")
        inspect_dataset_structure(dataset_path)
        raise

    print("\n===== STEP 3: EXTRACT FEATURES =====")
    features = extract_features(
        image_paths,
        batch_size=32,
        cache_path=os.path.join(output_dir, "resnet18_features.npy"),
    )

    print("\n===== STEP 4: BUILD FULL GRAPH =====")
    full_graph = build_full_graph(
        features=features,
        labels=labels,
        image_paths=image_paths,
        class_names=class_names,
        k=full_graph_k,
    )

    print("\n===== STEP 5: VISUALIZE DATASET / GRAPH =====")
    plot_class_distribution(labels, class_names, output_dir=output_dir)
    plot_degree_distribution(full_graph, output_dir=output_dir)
    plot_similarity_distribution(full_graph, output_dir=output_dir)
    visualize_graph(full_graph, image_paths, max_nodes=36, output_dir=output_dir)

    print("\n===== STEP 6: PREPARE FEW-SHOT SPLITS =====")
    split_info = prepare_fewshot_splits(
        labels=labels,
        class_names=class_names,
        k_shot=k_shot,
        q_query=q_query,
        seed=seed,
    )

    effective_n_way = min(
        n_way,
        max(2, len(split_info["train_classes"])),
        max(2, len(split_info["eligible_classes"])),
    )
    print(f"Using n_way = {effective_n_way}, k_shot = {k_shot}, q_query = {q_query}")

    print("\n===== STEP 7: TRAIN FEW-SHOT GNN =====")
    model = FewShotGNN(
        input_dim=full_graph.x.shape[1],
        hidden_dim=256,
        output_dim=128,
        dropout=0.3,
    )
    model, history, test_results = train_fewshot_model(
        model=model,
        full_graph=full_graph,
        split_info=split_info,
        epochs=epochs,
        episodes_per_epoch=episodes_per_epoch,
        val_episodes=val_episodes,
        test_episodes=test_episodes,
        n_way=effective_n_way,
        episode_graph_k=episode_graph_k,
        lr=1e-3,
        weight_decay=1e-4,
        temperature=0.1,
        output_dir=output_dir,
        seed=seed,
    )

    print("\n===== STEP 8: t-SNE BEFORE / AFTER GNN =====")
    model.eval()
    device = next(model.parameters()).device
    graph_device = full_graph.to(device)
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

    print("\n===== STEP 9: RUN GNN EXPLAINER =====")
    try:
        node_mask, edge_mask, explained_node = run_gnn_explainer(
            model,
            full_graph,
            node_idx=None,
            output_dir=output_dir,
            temperature=0.1,
        )
    except Exception as e:
        print(f"GNNExplainer failed: {e}")
        edge_mask = None

    print("\n===== STEP 10: INTERACTIVE GRAPH =====")
    try:
        visualize_interactive_graph(full_graph, image_paths, edge_mask=edge_mask, output_dir=output_dir)
    except Exception as e:
        print(f"Interactive graph failed: {e}")

    print("\nPipeline completed successfully.")
    print(f"All outputs saved in: {output_dir}")


if __name__ == "__main__":
    main()
