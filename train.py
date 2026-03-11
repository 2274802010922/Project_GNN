import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

def train(model, graphs, epochs=30):

    loader = DataLoader(graphs, batch_size=8, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    model.train()

    for epoch in range(epochs):

        total_loss = 0

        for data in loader:

            optimizer.zero_grad()

            out = model(data.x.float(), data.edge_index, data.batch)

            loss = F.cross_entropy(out, data.y)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
