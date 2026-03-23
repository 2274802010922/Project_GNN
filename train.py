import copy
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix

from graph_builder import build_class_index, build_episode_graph


def prepare_fewshot_splits(labels, class_names, k_shot=1, q_query=1, seed=42):
    """
    Prepare class splits for few-shot episodic training.
    If there are too few eligible classes, fallback to shared-class evaluation.
    """
    labels = np.asarray(labels, dtype=np.int64)
    class_to_indices = build_class_index(labels)
    min_required = k_shot + q_query

    eligible_classes = sorted(
        [cls for cls, indices in class_to_indices.items() if len(indices) >= min_required]
    )
    excluded_classes = sorted(set(class_to_indices.keys()) - set(eligible_classes))

    if len(eligible_classes) < 2:
        raise ValueError(
            f"Few-shot learning needs at least 2 classes with >= {min_required} samples each. "
            f"Eligible classes found: {len(eligible_classes)}"
        )

    rng = np.random.default_rng(seed)
    shuffled = np.array(eligible_classes, dtype=np.int64)
    rng.shuffle(shuffled)

    if len(eligible_classes) >= 6:
        n_total = len(shuffled)
        n_train = max(2, int(round(0.6 * n_total)))
        n_val = max(2, int(round(0.2 * n_total)))
        if n_train + n_val >= n_total:
            n_val = max(1, n_total - n_train - 1)
        train_classes = shuffled[:n_train].tolist()
        val_classes = shuffled[n_train:n_train + n_val].tolist()
        test_classes = shuffled[n_train + n_val:].tolist()

        if len(test_classes) < 2 or len(val_classes) < 2:
            split_mode = "shared_class"
            train_classes = eligible_classes
            val_classes = eligible_classes
            test_classes = eligible_classes
        else:
            split_mode = "class_disjoint"
    else:
        split_mode = "shared_class"
        train_classes = eligible_classes
        val_classes = eligible_classes
        test_classes = eligible_classes

    split_info = {
        "split_mode": split_mode,
        "eligible_classes": eligible_classes,
        "excluded_classes": excluded_classes,
        "eligible_class_names": [class_names[c] for c in eligible_classes],
        "train_classes": train_classes,
        "val_classes": val_classes,
        "test_classes": test_classes,
        "class_to_indices": class_to_indices,
        "k_shot": k_shot,
        "q_query": q_query,
        "seed": seed,
    }

    print(f"Few-shot split mode: {split_mode}")
    print(f"Eligible classes: {len(eligible_classes)}")
    if excluded_classes:
        print(
            f"Excluded classes (< {min_required} samples): {len(excluded_classes)} -> "
            f"{[class_names[c] for c in excluded_classes[:10]]}"
        )
    print(f"Train classes: {len(train_classes)}")
    print(f"Val classes: {len(val_classes)}")
    print(f"Test classes: {len(test_classes)}")

    return split_info


def sample_episode(split_info, split_name, n_way=5, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    available_classes = split_info[f"{split_name}_classes"]
    if len(available_classes) < 2:
        raise ValueError(f"Need at least 2 classes in {split_name} split to sample an episode.")

    n_way = min(n_way, len(available_classes))
    selected_classes = rng.choice(np.array(available_classes), size=n_way, replace=False)

    support_indices = []
    query_indices = []
    k_shot = split_info["k_shot"]
    q_query = split_info["q_query"]

    for cls in selected_classes:
        pool = split_info["class_to_indices"][int(cls)]
        sample_size = k_shot + q_query
        chosen = rng.choice(pool, size=sample_size, replace=False)
        support_indices.extend(chosen[:k_shot].tolist())
        query_indices.extend(chosen[k_shot:].tolist())

    episode_indices = np.array(support_indices + query_indices, dtype=np.int64)
    support_mask = np.zeros(len(episode_indices), dtype=bool)
    query_mask = np.zeros(len(episode_indices), dtype=bool)
    support_mask[:len(support_indices)] = True
    query_mask[len(support_indices):] = True

    return {
        "episode_indices": episode_indices,
        "support_mask": torch.tensor(support_mask, dtype=torch.bool),
        "query_mask": torch.tensor(query_mask, dtype=torch.bool),
        "selected_classes": np.array(selected_classes, dtype=np.int64),
    }


def compute_prototypes(embeddings, labels, num_classes):
    prototypes = []
    for class_id in range(num_classes):
        class_embeddings = embeddings[labels == class_id]
        prototypes.append(class_embeddings.mean(dim=0))
    prototypes = torch.stack(prototypes, dim=0)
    prototypes = F.normalize(prototypes, p=2, dim=1)
    return prototypes


def episode_forward(model, episode_data, support_mask, query_mask, temperature=0.1):
    embeddings = model(episode_data.x, episode_data.edge_index, episode_data.edge_weight)
    support_emb = embeddings[support_mask]
    query_emb = embeddings[query_mask]
    support_y = episode_data.y[support_mask]
    query_y = episode_data.y[query_mask]

    prototypes = compute_prototypes(
        embeddings=support_emb,
        labels=support_y,
        num_classes=int(episode_data.episode_class_ids.numel()),
    )

    logits = torch.mm(query_emb, prototypes.t()) / temperature
    loss = F.cross_entropy(logits, query_y)
    preds = logits.argmax(dim=1)
    acc = float((preds == query_y).float().mean().item())
    return loss, acc, preds, query_y, embeddings


def plot_training_history(history, save_path=None):
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Episode Loss")
    if history["val_loss"]:
        plt.plot(epochs, history["val_loss"], label="Val Episode Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Few-Shot Episodic Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Episode Acc")
    if history["val_acc"]:
        plt.plot(epochs, history["val_acc"], label="Val Episode Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Few-Shot Episodic Accuracy")
    plt.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    labels = np.arange(len(class_names))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    figsize = (max(8, 0.45 * len(class_names)), max(6, 0.45 * len(class_names)))
    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation="nearest")
    plt.title("Few-Shot Test Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] != 0:
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


def evaluate_fewshot(
    model,
    features,
    global_labels,
    split_info,
    split_name,
    n_way=5,
    num_episodes=50,
    episode_graph_k=3,
    temperature=0.1,
    device=None,
    seed=123,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    rng = np.random.default_rng(seed)

    losses = []
    accuracies = []
    y_true_global = []
    y_pred_global = []

    with torch.no_grad():
        for _ in range(num_episodes):
            episode = sample_episode(split_info, split_name=split_name, n_way=n_way, rng=rng)
            episode_data = build_episode_graph(
                features=features,
                global_labels=global_labels,
                episode_indices=episode["episode_indices"],
                episode_class_ids=episode["selected_classes"],
                k=episode_graph_k,
            ).to(device)

            support_mask = episode["support_mask"].to(device)
            query_mask = episode["query_mask"].to(device)

            loss, acc, preds_local, query_y_local, _ = episode_forward(
                model=model,
                episode_data=episode_data,
                support_mask=support_mask,
                query_mask=query_mask,
                temperature=temperature,
            )
            losses.append(float(loss.item()))
            accuracies.append(acc)

            selected_classes = episode["selected_classes"]
            y_true_global.extend(selected_classes[query_y_local.cpu().numpy()].tolist())
            y_pred_global.extend(selected_classes[preds_local.cpu().numpy()].tolist())

    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "acc": float(np.mean(accuracies)) if accuracies else float("nan"),
        "y_true_global": np.array(y_true_global, dtype=np.int64),
        "y_pred_global": np.array(y_pred_global, dtype=np.int64),
    }


def train_fewshot_model(
    model,
    full_graph,
    split_info,
    epochs=40,
    episodes_per_epoch=30,
    val_episodes=20,
    test_episodes=50,
    n_way=5,
    episode_graph_k=3,
    lr=1e-3,
    weight_decay=1e-4,
    temperature=0.1,
    output_dir="outputs",
    seed=42,
):
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    features = full_graph.raw_features.cpu().numpy()
    global_labels = full_graph.y.cpu().numpy()

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_state = copy.deepcopy(model.state_dict())
    best_val_acc = -1.0

    train_rng = np.random.default_rng(seed)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        epoch_accs = []

        for _ in range(episodes_per_epoch):
            episode = sample_episode(split_info, split_name="train", n_way=n_way, rng=train_rng)
            episode_data = build_episode_graph(
                features=features,
                global_labels=global_labels,
                episode_indices=episode["episode_indices"],
                episode_class_ids=episode["selected_classes"],
                k=episode_graph_k,
            ).to(device)

            support_mask = episode["support_mask"].to(device)
            query_mask = episode["query_mask"].to(device)

            optimizer.zero_grad()
            loss, acc, _, _, _ = episode_forward(
                model=model,
                episode_data=episode_data,
                support_mask=support_mask,
                query_mask=query_mask,
                temperature=temperature,
            )
            loss.backward()
            optimizer.step()

            epoch_losses.append(float(loss.item()))
            epoch_accs.append(acc)

        train_loss = float(np.mean(epoch_losses))
        train_acc = float(np.mean(epoch_accs))

        val_results = evaluate_fewshot(
            model=model,
            features=features,
            global_labels=global_labels,
            split_info=split_info,
            split_name="val",
            n_way=n_way,
            num_episodes=val_episodes,
            episode_graph_k=episode_graph_k,
            temperature=temperature,
            device=device,
            seed=seed + epoch,
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_results["loss"])
        history["val_acc"].append(val_results["acc"])

        metric_for_selection = val_results["acc"]
        if np.isnan(metric_for_selection):
            metric_for_selection = train_acc

        if metric_for_selection >= best_val_acc:
            best_val_acc = metric_for_selection
            best_state = copy.deepcopy(model.state_dict())

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(
                f"Epoch {epoch:03d} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_results['loss']:.4f} | Val Acc: {val_results['acc']:.4f}"
            )

    model.load_state_dict(best_state)

    weight_path = os.path.join(output_dir, "fewshot_gnn.pt")
    torch.save(model.state_dict(), weight_path)
    print(f"Saved model weights to: {weight_path}")

    plot_training_history(history, save_path=os.path.join(output_dir, "training_curves.png"))

    test_results = evaluate_fewshot(
        model=model,
        features=features,
        global_labels=global_labels,
        split_info=split_info,
        split_name="test",
        n_way=n_way,
        num_episodes=test_episodes,
        episode_graph_k=episode_graph_k,
        temperature=temperature,
        device=device,
        seed=seed + 999,
    )

    print("\nFew-shot evaluation summary:")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Test episodic loss: {test_results['loss']:.4f}")
    print(f"Test episodic accuracy: {test_results['acc']:.4f}")

    present_classes = sorted(set(test_results["y_true_global"].tolist()) | set(test_results["y_pred_global"].tolist()))
    if present_classes:
        report_names = [full_graph.class_names[idx] for idx in present_classes]
        print("\nClassification report on aggregated test queries:")
        print(
            classification_report(
                test_results["y_true_global"],
                test_results["y_pred_global"],
                labels=present_classes,
                target_names=report_names,
                zero_division=0,
            )
        )

        plot_confusion_matrix(
            y_true=test_results["y_true_global"],
            y_pred=test_results["y_pred_global"],
            class_names=full_graph.class_names,
            save_path=os.path.join(output_dir, "confusion_matrix.png"),
        )

    class_counter = Counter(global_labels.tolist())
    print("\nClass counts:")
    for class_id, count in class_counter.most_common(15):
        print(f"  {full_graph.class_names[class_id]}: {count}")

    return model, history, test_results
