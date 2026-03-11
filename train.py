import torch
import torch.nn.functional as F


def train_model(model, data, epochs=50):

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    y = torch.randint(0, 10, (data.x.shape[0],))

    for epoch in range(epochs):

        optimizer.zero_grad()

        out = model(data.x, data.edge_index)

        loss = F.cross_entropy(out, y)

        loss.backward()

        optimizer.step()

        print(f"Epoch {epoch+1} Loss {loss.item():.4f}")
