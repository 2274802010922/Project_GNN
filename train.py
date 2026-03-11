import torch

def train_model(model, train_loader, test_loader, epochs=50, lr=0.001):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):

        model.train()
        total_loss = 0

        for data in train_loader:

            optimizer.zero_grad()

            out = model(data.x, data.edge_index, data.batch)

            loss = criterion(out, data.y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        acc = evaluate(model, test_loader)

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Test Acc: {acc:.4f}")

    return model


def evaluate(model, loader):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for data in loader:

            out = model(data.x, data.edge_index, data.batch)

            pred = out.argmax(dim=1)

            correct += (pred == data.y).sum().item()

            total += data.y.size(0)

    return correct / total
