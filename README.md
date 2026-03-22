Visualization and Results
Training Pipeline
<p align="center"> <img src="images/pipeline.png" width="700"> </p>

The pipeline includes dataset loading, feature extraction, graph construction, and GNN training. The constructed graph consists of 99 nodes and 8 edges, indicating a sparse structure.

Training Loss (Early Stage)
<p align="center"> <img src="images/loss_early.png" width="600"> </p>

The loss decreases steadily in the early training stage, showing stable learning behavior and effective gradient updates.

Training Loss (Late Stage)
<p align="center"> <img src="images/loss_late.png" width="600"> </p>

The loss converges to approximately 0.14, demonstrating strong model convergence and effective optimization.

GNN Explainer Scores
<p align="center"> <img src="images/explainer_scores.png" width="700"> </p>

The explainer highlights important nodes and edges. A subset of edges has significantly higher importance, indicating that the model focuses on meaningful relationships.

Explanation Graph
<p align="center"> <img src="images/explanation_graph.png" width="600"> </p>

This visualization shows the most influential nodes contributing to the prediction, improving model interpretability.

Graph Visualization with Images
<p align="center"> <img src="images/graph_with_images.png" width="700"> </p>

Each node corresponds to a real image, allowing intuitive understanding of how data is structured in the graph.

Interactive Graph
<p align="center"> <img src="images/interactive_graph.png" width="700"> </p>

The interactive graph enables dynamic exploration of node relationships and enhances understanding of the graph structure.
