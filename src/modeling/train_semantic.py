import os
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from src.features.semantic_extractor import SemanticClassifier
from src.preprocessing.image_loader import get_dataloaders

CHECKPOINT_PATH = "saved_models/semantic_checkpoint.pth"


def compute_class_weights(dataset, num_classes=2, device="cpu"):
    """
    Inverse-frequency class weights for CrossEntropyLoss, computed from the
    TRAINING set only (never val/test).

    Celeb-DF v2 is heavily imbalanced (~86% fake / ~14% real). Without this,
    CrossEntropyLoss has no penalty for a model that just learns to predict
    "fake" almost always — that shortcut minimizes average loss on this
    dataset without learning to discriminate real videos at all, which is
    exactly the failure mode observed in an earlier unweighted run (recall
    for "real" collapsed to ~0.03 despite ~86% overall accuracy).

    weight_c = total_samples / (num_classes * count_c)
    — the standard "balanced" weighting (same formula sklearn's
    class_weight='balanced' uses), giving the minority class a
    proportionally larger gradient contribution per mistake.
    """
    label_counts = Counter(label for _, label in dataset.samples)
    total = sum(label_counts.values())

    weights = [
        total / (num_classes * label_counts.get(c, 1))
        for c in range(num_classes)
    ]

    print(f"Class counts (train set): {dict(label_counts)}")
    print(f"Class weights applied: {dict(enumerate(round(w, 3) for w in weights))}")

    return torch.tensor(weights, dtype=torch.float32, device=device)


@torch.no_grad()
def evaluate_metrics(model, loader, device):
    """
    Full validation pass returning accuracy, macro-F1, and balanced
    accuracy. Macro-F1 (average of each class's own F1 score, unweighted by
    class frequency) is used for model selection instead of raw accuracy,
    since raw accuracy on an imbalanced dataset can keep improving purely by
    leaning harder into the majority class — which is what happened before
    this fix. Balanced accuracy (average per-class recall) is reported
    alongside as an easy sanity check: for a model that's collapsed to
    predicting one class, this drops to ~0.50.
    """
    model.eval()
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        _, logits = model(images)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())

    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)

    # sklearn returns numpy.float64 scalars — cast to native Python float so
    # these are safe to torch.save() into a checkpoint. PyTorch 2.6+
    # tightened torch.load's default (weights_only=True) to reject
    # unpickling numpy scalar types, which would otherwise break loading
    # this exact checkpoint later.
    return float(accuracy), float(macro_f1), float(balanced_acc)


def main():
    os.makedirs("saved_models", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train + validation only — the test set is intentionally not loaded
    # here. Model selection must never see it. See test_semantic.py for the
    # one-time final evaluation.
    train_loader, val_loader, _ = get_dataloaders(batch_size=16)

    class_weights = compute_class_weights(train_loader.dataset, device=device)

    model = SemanticClassifier().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scaler = GradScaler()

    epochs = 5
    best_val_macro_f1 = -1.0  # ensures epoch 1 always saves a checkpoint, even in a degenerate all-zero-F1 edge case

    for epoch in range(epochs):
        model.train()

        total_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            with autocast():
                _, logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_acc = train_correct / train_total if train_total > 0 else 0.0
        val_acc, val_macro_f1, val_balanced_acc = evaluate_metrics(model, val_loader, device)

        print(
            f"Epoch [{epoch + 1}/{epochs}] | "
            f"Train Loss: {total_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val Macro-F1: {val_macro_f1:.4f} | "
            f"Val Balanced Acc: {val_balanced_acc:.4f}"
        )

        # Model selection uses validation MACRO-F1, not raw accuracy — see
        # evaluate_metrics()'s docstring for why. The test set is never
        # touched during training either way.
        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1

            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch + 1,
                "val_accuracy": val_acc,
                "val_macro_f1": val_macro_f1,
                "val_balanced_accuracy": val_balanced_acc,
            }, CHECKPOINT_PATH)

            print(f"  -> New best checkpoint saved (Val Macro-F1: {val_macro_f1:.4f})")

    print("\nTraining complete.")
    print(f"Best Validation Macro-F1: {best_val_macro_f1:.4f}")
    print(f"Checkpoint saved to: {CHECKPOINT_PATH}")
    print("\nRun `python -m src.modeling.test_semantic` for the final, "
          "one-time test set evaluation.")


if __name__ == "__main__":
    main()
