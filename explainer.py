import torch
from torch_geometric.explain import Explainer, GNNExplainer

def run_explainer(model, data, reverse_label_map, device):

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

    data = data.to(device)

    model.eval()

    with torch.no_grad():
        embeddings = model(data.x, data.edge_index)

    unique_labels = data.y.unique()

    prototypes = []

    for lab in unique_labels:

        mask = data.y == lab
        proto = embeddings[mask].mean(dim=0)
        prototypes.append(proto)

    prototypes = torch.stack(prototypes)

    dists = torch.cdist(embeddings, prototypes)

    pred = dists.argmin(dim=1)

    pred_class = unique_labels[pred[0]].item()
    true_class = data.y[0].item()

    print("Predicted:", reverse_label_map[pred_class])
    print("Ground truth:", reverse_label_map[true_class])

    explanation = explainer(
        x=data.x,
        edge_index=data.edge_index,
        index=0,
    )

    return explanation