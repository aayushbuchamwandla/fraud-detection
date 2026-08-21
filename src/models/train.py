"""
Train the FraudMLP classifier and evaluate it with metrics appropriate for
a highly imbalanced binary classification problem.

Why not accuracy: with 0.17% fraud, a model that predicts "not fraud" for
every transaction scores ~99.83% accuracy while catching zero fraud. Accuracy
is meaningless here. Instead this script reports:

  - Precision: of transactions flagged as fraud, how many actually are.
    Low precision means too many legitimate customers get blocked/reviewed.
  - Recall: of actual fraud, how much was caught. Low recall means fraud
    slips through undetected -- usually the costlier failure mode.
  - F1: harmonic mean of precision/recall, single number for model comparison.
  - ROC-AUC: ranking quality across all thresholds, but can look optimistic
    on imbalanced data because true negatives dominate the denominator.
  - PR-AUC (average precision): more informative than ROC-AUC here because it
    ignores the (huge, uninteresting) true-negative count and focuses on how
    well the model ranks fraud above non-fraud.
  - Confusion matrix: the actual counts behind the above, at a chosen
    decision threshold.

Class imbalance handling: BCEWithLogitsLoss with a computed pos_weight
(ratio of negative to positive examples in the TRAIN split) upweights the
rare fraud class in the loss, rather than resampling/duplicating the data.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data.dataset import FraudDataset, get_dataloader
from src.models.model import FraudMLP, ModelConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"

RANDOM_SEED = 42


def set_seed(seed: int = RANDOM_SEED) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def compute_pos_weight(dataset: FraudDataset) -> torch.Tensor:
    n_pos = int(dataset.y.sum().item())
    n_neg = len(dataset) - n_pos
    return torch.tensor(n_neg / n_pos, dtype=torch.float32)


@torch.no_grad()
def get_probs_labels(model: nn.Module, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs, all_labels = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        probs = model.predict_proba(xb).cpu()
        all_probs.append(probs)
        all_labels.append(yb)
    return torch.cat(all_probs).numpy(), torch.cat(all_labels).numpy()


def best_f1_threshold(probs: np.ndarray, labels: np.ndarray) -> float:
    """Find the decision threshold that maximizes F1 on the given (val) set."""
    precisions, recalls, thresholds = precision_recall_curve(labels, probs)
    f1s = np.where(
        (precisions + recalls) > 0,
        2 * precisions * recalls / (precisions + recalls + 1e-12),
        0.0,
    )
    # precision_recall_curve returns len(thresholds) == len(precisions) - 1
    best_idx = int(np.argmax(f1s[:-1]))
    return float(thresholds[best_idx])


def metrics_at_threshold(probs: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    preds = (probs >= threshold).astype(int)

    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "threshold": threshold,
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "roc_auc": roc_auc_score(labels, probs),
        "pr_auc": average_precision_score(labels, probs),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n_samples": len(labels),
        "n_fraud": int(labels.sum()),
    }


def train(
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    device_str: str = "cpu",
) -> dict:
    set_seed(RANDOM_SEED)
    device = torch.device(device_str)

    train_loader = get_dataloader("train", batch_size=batch_size)
    val_loader = get_dataloader("val", batch_size=batch_size, shuffle=False)
    train_ds = train_loader.dataset

    model = FraudMLP(ModelConfig(input_dim=train_ds.n_features)).to(device)
    pos_weight = compute_pos_weight(train_ds).to(device)
    print(f"pos_weight (neg/pos ratio in train): {pos_weight.item():.2f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = []
    best_val_f1 = -1.0
    best_state = None

    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        val_probs, val_labels = get_probs_labels(model, val_loader, device)
        val_metrics = metrics_at_threshold(val_probs, val_labels, threshold=0.5)
        history.append(
            {
                "epoch": epoch,
                "train_loss": epoch_loss / n_batches,
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"],
                "val_roc_auc": val_metrics["roc_auc"],
                "val_pr_auc": val_metrics["pr_auc"],
            }
        )
        print(
            f"epoch {epoch:2d}/{epochs} | loss {history[-1]['train_loss']:.4f} | "
            f"val P {val_metrics['precision']:.3f} R {val_metrics['recall']:.3f} "
            f"F1 {val_metrics['f1']:.3f} ROC-AUC {val_metrics['roc_auc']:.4f} "
            f"PR-AUC {val_metrics['pr_auc']:.4f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    train_seconds = time.perf_counter() - start

    model.load_state_dict(best_state)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT_DIR / "fraud_mlp.pt")
    model.config.save(CHECKPOINT_DIR / "fraud_mlp_config.json")

    # Select the decision threshold on VAL (never on test) by maximizing F1,
    # then apply that fixed threshold to the held-out test set.
    val_probs, val_labels = get_probs_labels(model, val_loader, device)
    tuned_threshold = best_f1_threshold(val_probs, val_labels)

    test_loader = get_dataloader("test", batch_size=batch_size, shuffle=False)
    test_probs, test_labels = get_probs_labels(model, test_loader, device)
    test_metrics_default = metrics_at_threshold(test_probs, test_labels, threshold=0.5)
    test_metrics_tuned = metrics_at_threshold(test_probs, test_labels, threshold=tuned_threshold)

    results = {
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "device": device_str,
        "train_seconds": round(train_seconds, 2),
        "best_val_f1": best_val_f1,
        "tuned_threshold": tuned_threshold,
        "history": history,
        "test_metrics_default_threshold": test_metrics_default,
        "test_metrics_tuned_threshold": test_metrics_tuned,
    }
    with open(CHECKPOINT_DIR / "training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Persist the tuned threshold alongside the model so inference code uses it
    with open(CHECKPOINT_DIR / "fraud_mlp_config.json") as f:
        config_dict = json.load(f)
    config_dict["decision_threshold"] = tuned_threshold
    with open(CHECKPOINT_DIR / "fraud_mlp_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    print(f"\nVal-tuned decision threshold (maximizes F1 on val): {tuned_threshold:.4f}")
    print("\n=== Test set @ threshold=0.5 (default) ===")
    print(json.dumps(test_metrics_default, indent=2))
    print(f"\n=== Test set @ threshold={tuned_threshold:.4f} (tuned on val) ===")
    print(json.dumps(test_metrics_tuned, indent=2))
    print(f"\nModel saved to {CHECKPOINT_DIR / 'fraud_mlp.pt'}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device_str=args.device)
