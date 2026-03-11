import torch
import torch.nn.functional as F


def prototype_loss(embeddings,labels):

    unique_labels = labels.unique()

    prototypes = []

    for lab in unique_labels:

        mask = labels == lab

        proto = embeddings[mask].mean(dim=0)

        prototypes.append(proto)

    prototypes = torch.stack(prototypes)

    dists = torch.cdist(embeddings,prototypes)

    label_map = {lab.item():i for i,lab in enumerate(unique_labels)}

    new_labels = torch.tensor(
        [label_map[l.item()] for l in labels]
    ).to(embeddings.device)

    loss = F.cross_entropy(-dists,new_labels)

    return loss


def train(model,loader,optimizer,device):

    model.train()

    total_loss = 0

    for data in loader:

        data = data.to(device)

        optimizer.zero_grad()

        embeddings = model(data.x,data.edge_index)

        loss = prototype_loss(embeddings,data.y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)
