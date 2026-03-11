from torch_geometric.explain import Explainer,GNNExplainer


def create_explainer(model):

    explainer = Explainer(

        model=model,

        algorithm=GNNExplainer(epochs=200),

        explanation_type='model',

        node_mask_type='attributes',

        edge_mask_type='object',

        model_config=dict(

            mode='multiclass_classification',

            task_level='node',

            return_type='raw',

        ),
    )

    return explainer
