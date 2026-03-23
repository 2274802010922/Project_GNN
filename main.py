import os
import random

import numpy as np
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



def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def main():
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    seed = 42
    set_seed(seed)

    # Tuned configuration for better accuracy while keeping the same few-shot + graph idea.
    k_shot = 1
    q_query = 1
    n_way = 5
    full_graph_k = 7
    episode_graph_k = 4
    epochs = 80
    episodes_per_epoch = 100
    val_episodes = 60
    test_episodes = 100

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

    print(f"Loaded label source: {label_source}")
    print(f"Num samples: {len(image_paths)} | Num classes: {len(class_names)}")

    print("\n===== STEP 3: EXTRACT FEATURES =====")
    features = extract_features(
        image_paths,
        batch_size=16,
        cache_path=os.path.join(output_dir, "resnet50_tta_features.npy"),
        model_name="resnet50",
        tta=True,
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
        hidden_dim=384,
        output_dim=192,
        dropout=0.35,
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
        lr=7e-4,
        weight_decay=1e-4,
        temperature=0.12,
        output_dir=output_dir,
        seed=seed,
        patience=15,
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
        _, edge_mask, _ = run_gnn_explainer(
            model,
            full_graph,
            node_idx=None,
            output_dir=output_dir,
            temperature=0.12,
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
