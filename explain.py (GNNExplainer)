from torch_geometric.explain import Explainer, GNNExplainer


def run_explainer(model, data):

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=100),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object'
    )

    explanation = explainer(data.x, data.edge_index)

    print("Explanation generated")

    return explanation
