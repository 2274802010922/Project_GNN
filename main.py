from dataset import download_dataset
from feature_extractor import extract_features
from graph_builder import build_graph
from gnn_model import GNN
from train import train_model
from explain import run_explainer
from visualize import visualize_graph


def main():

    print("\n===== STEP 1: DOWNLOAD DATASET =====")
    dataset_path = download_dataset()

    print("\n===== STEP 2: EXTRACT FEATURES =====")
    features = extract_features(dataset_path)

    print("\n===== STEP 3: BUILD GRAPH =====")
    graph = build_graph(features)

    print("\n===== STEP 4: TRAIN GNN =====")
    model = GNN(graph.x.shape[1])
    train_model(model, graph)

    print("\n===== STEP 5: RUN GNN EXPLAINER =====")
    explanation = run_explainer(model, graph)

    print("\n===== STEP 6: VISUALIZE GRAPH =====")
    visualize_graph(graph)


if __name__ == "__main__":
    main()
