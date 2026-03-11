from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.explain.config import ModelConfig

def run_explainer(model, data):

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type='model',
        model_config=ModelConfig(
            mode='classification',
            task_level='graph',
            return_type='raw'
        ),
    )

    explanation = explainer(
        x=data.x,
        edge_index=data.edge_index
    )

    return explanation
