Results and Visualization
Training Pipeline
<p align="center"> <img src="images/pipeline.png" width="700"> </p>

The pipeline includes dataset loading, feature extraction, graph construction, and Graph Neural Network training. The constructed graph consists of 99 nodes and 8 edges, indicating a sparse structure.

Training Loss (Early Stage)
<p align="center"> <img src="images/loss_early.png" width="600"> </p>

The loss decreases steadily during the early training stage, indicating stable learning behavior and effective gradient updates. The model quickly learns basic patterns from the data.

Training Loss (Late Stage)
<p align="center"> <img src="images/loss_late.png" width="600"> </p>

The loss continues to decrease and converges to approximately 0.14. This demonstrates strong model convergence and effective optimization, with the model fine-tuning its parameters in later epochs.

GNN Explainer Scores
<p align="center"> <img src="images/explainer_scores.png" width="700"> </p>

The explainer identifies important nodes and edges. A subset of edges has significantly higher importance, while others contribute minimally. This shows that the model focuses on meaningful relationships and ignores irrelevant connections.

Explanation Graph
<p align="center"> <img src="images/explanation_graph.png" width="600"> </p>

This graph highlights the most influential nodes contributing to the prediction. It improves model interpretability by showing how different nodes affect the output.

Graph Visualization with Images
<p align="center"> <img src="images/graph_with_images.png" width="700"> </p>

Each node corresponds to a real image, allowing intuitive understanding of how visual data is structured within the graph. This bridges the gap between raw data and graph representation.

Interactive Graph
<p align="center"> <img src="images/interactive_graph.png" width="700"> </p>

The interactive graph enables dynamic exploration of node relationships. Users can better understand graph structure and connectivity.

Overall Analysis

The model demonstrates strong performance despite the sparse graph structure. The training process is stable, and the final loss value indicates effective learning.

The GNNExplainer results show that the model selectively focuses on important edges, confirming its ability to capture meaningful structural relationships.

Visualization results further validate that the model successfully learns both feature-level and graph-level representations, while also providing interpretability.
