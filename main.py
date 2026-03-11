import torch
from dataset import build_graph_dataset, build_dataloaders
from model import GCN
from train import train,test
from explainer import run_explainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

graphs,label_map = build_graph_dataset()

train_loader, test_loader = build_dataloaders(graphs)
input_dim = graphs[0].x.shape[1]

model = GCN(input_dim).to(device)

optimizer = torch.optim.Adam(model.parameters(),lr=0.0005)

best_acc = 0

for epoch in range(50):

    loss = train(model,train_loader,optimizer)

    acc = test(model,test_loader)

    print(f"Epoch {epoch+1}, Loss: {loss:.4f}, Test Acc: {acc:.4f}")

    if acc > best_acc:

        best_acc = acc
        torch.save(model.state_dict(),"best_model.pth")

print("Best Accuracy:",best_acc)

model.load_state_dict(torch.load("best_model.pth"))

data = test_graphs[0].to(device)

run_explainer(model,data)

