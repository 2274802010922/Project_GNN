import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix


def _accuracy(logits, labels):
    preds = logits.argmax(dim=1)
    return float((preds == labels).float().mean().item())


def plot_training_history(history, save_path=None):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    if len(history["val_loss"]) > 0:
        plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / Validation Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    if len(history["val_acc"]) > 0:
        plt.plot(epochs, history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training / Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.0 if cm.size > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def train_model(model, data, epochs=80, lr=1e-3, weight_decay=1e-4, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    data = data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    best_val_acc = -1.0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(data.x, data.edge_index, data.edge_weight)
        train_loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            eval_logits = model(data.x, data.edge_index, data.edge_weight)
            train_acc = _accuracy(eval_logits[data.train_mask], data.y[data.train_mask])

            if int(data.val_mask.sum()) > 0:
                val_loss = F.cross_entropy(eval_logits[data.val_mask], data.y[data.val_mask]).item()
                val_acc = _accuracy(eval_logits[data.val_mask], data.y[data.val_mask])
            else:
                val_loss = float("nan")
                val_acc = float("nan")

        history["train_loss"].append(float(train_loss.item()))
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if not np.isnan(val_acc) and val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"Epoch {epoch:03d} | "
                f"Train Loss: {train_loss.item():.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            )

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        final_logits = model(data.x, data.edge_index, data.edge_weight)
        preds = final_logits.argmax(dim=1)

    train_acc = _accuracy(final_logits[data.train_mask], data.y[data.train_mask])
    val_acc = (
        _accuracy(final_logits[data.val_mask], data.y[data.val_mask])
        if int(data.val_mask.sum()) > 0 else float("nan")
    )
    test_acc = (
        _accuracy(final_logits[data.test_mask], data.y[data.test_mask])
        if int(data.test_mask.sum()) > 0 else float("nan")
    )

    print("\nBest model performance:")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    plot_training_history(
        history,
        save_path=os.path.join(output_dir, "training_curves.png"),
    )

    test_predictions = None
    if int(data.test_mask.sum()) > 0:
        y_true = data.y[data.test_mask].cpu().numpy()
        y_pred = preds[data.test_mask].cpu().numpy()
        test_predictions = (y_true, y_pred)

        print("\nClassification report on test set:")
        print(classification_report(
            y_true,
            y_pred,
            target_names=data.class_names,
            zero_division=0,
        ))

        plot_confusion_matrix(
            y_true,
            y_pred,
            data.class_names,
            save_path=os.path.join(output_dir, "confusion_matrix.png"),
        )

    return model, history, final_logits.detach().cpu(), test_predictions
