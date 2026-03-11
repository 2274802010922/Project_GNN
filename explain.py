from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.explain.config import ModelConfig
import matplotlib.pyplot as plt


def run_explainer(model, data):

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=100),

        model_config=ModelConfig(
            mode="multiclass_classification",
            task_level="node",
            return_type="raw"
        ),

        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object"
    )

    explanation = explainer(data.x, data.edge_index)

    print("GNNExplainer finished")

    # print importance
    print("Top important edges:")
    print(explanation.edge_mask[:10])

    # visualize explanation
    explanation.visualize_graph()

    plt.show()

    return explanation
