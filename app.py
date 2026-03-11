import streamlit as st
from dataset import download_dataset
from feature_extractor import extract_features
from graph_builder import build_graph
from gnn_model import GNN
from train import train_model
from explain import run_gnn_explainer

st.title("GNN Image Graph Explorer")

st.write("Build graph from images and explain relationships using GNN")

if st.button("Run AI Pipeline"):

    st.write("STEP 1: Download dataset")
    dataset_path = download_dataset()

    st.write("STEP 2: Extract CNN features")
    features, image_paths = extract_features(dataset_path)

    st.write("STEP 3: Build graph")
    graph = build_graph(features)

    st.write("STEP 4: Train GNN")
    model = GNN(graph.x.shape[1])
    train_model(model, graph)

    st.write("STEP 5: Explain graph")
    node_mask, edge_mask = run_gnn_explainer(model, graph)

    st.success("Pipeline finished!")
