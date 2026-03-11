from torch_geometric.explain import Explainer, GNNExplainer

def explain(model, graph):

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=100),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='graph',
            return_type='raw'
        ),
    )

    explanation = explainer(graph.x, graph.edge_index)

    return explanation
