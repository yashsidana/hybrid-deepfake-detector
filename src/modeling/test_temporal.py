import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.features.temporal_extractor import TemporalClassifier
from src.preprocessing.temporal_dataset import get_dataloaders

CHECKPOINT_PATH = "saved_models/temporal_checkpoint.pth"
REPORT_PATH = "saved_models/test_temporal_evaluation_report.json"

CLASS_NAMES = ["real", "fake"]  # label 0 = real, label 1 = fake


def load_model(device):
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"No checkpoint found at {CHECKPOINT_PATH}. "
            f"Run `python -m src.modeling.train_temporal` first."
        )

    model = TemporalClassifier().to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(
        f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')} "
        f"(selected on Val Macro-F1: {checkpoint.get('val_macro_f1', float('nan')):.4f}, "
        f"Val Acc: {checkpoint.get('val_accuracy', float('nan')):.4f})"
    )
    return model


@torch.no_grad()
def run_inference(model, loader, device):
    all_preds = []
    all_labels = []
    all_fake_probs = []

    for sequences, labels in loader:
        sequences = sequences.to(device)
        _, logits = model(sequences)

        probs = F.softmax(logits, dim=1)
        fake_probs = probs[:, 1].cpu().numpy()  # P(fake)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())
        all_fake_probs.extend(fake_probs.tolist())

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_fake_probs),
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader = get_dataloaders(batch_size=8)
    model = load_model(device)

    all_labels, all_preds, all_fake_probs = run_inference(model, test_loader, device)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    report = classification_report(
        all_labels, all_preds, labels=[0, 1], target_names=CLASS_NAMES, zero_division=0
    )

    # ROC-AUC needs both classes present in the test set to be defined.
    if len(np.unique(all_labels)) == 2:
        roc_auc = roc_auc_score(all_labels, all_fake_probs)
    else:
        roc_auc = None
        print(
            "\n[Warning] ROC-AUC skipped: test set contains only one class. "
            "Check data/splits/test.csv."
        )

    print("\n" + "=" * 55)
    print("TEMPORAL BRANCH — FINAL TEST SET EVALUATION")
    print("=" * 55)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}" if roc_auc is not None else "ROC-AUC:   N/A")
    print(f"\nConfusion Matrix (rows=true, cols=pred) {CLASS_NAMES}:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
        }, f, indent=2)

    print(f"\nSaved evaluation report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
