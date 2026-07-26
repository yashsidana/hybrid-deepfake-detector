import logging
import os
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from src.features.temporal_extractor import TemporalClassifier
from src.preprocessing.temporal_dataset import get_dataloaders

CHECKPOINT_PATH = "saved_models/temporal_checkpoint.pth"
LOG_PATH = "saved_models/train_temporal.log"

EPOCHS = 20
EARLY_STOP_PATIENCE = 5
LR = 1e-3
BATCH_SIZE = 8

# Must exist before logging.FileHandler(LOG_PATH) below runs at import time.
os.makedirs("saved_models", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="a"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def compute_class_weights(dataset, num_classes=2, device="cpu"):
    """
    Inverse-frequency class weights for CrossEntropyLoss, computed from the
    TRAINING set only. Same imbalance problem as the semantic branch (~86%
    fake / ~14% real in Celeb-DF v2) applies identically here — without
    this, the model can minimize loss by learning to predict "fake" almost
    always, which looks like high accuracy but means it never learned to
    recognize real videos. See train_semantic.py for the fuller rationale
    and a worked example of this exact failure mode.
    """
    label_counts = Counter(label for _, label in dataset.samples)
    total = sum(label_counts.values())

    weights = [
        total / (num_classes * label_counts.get(c, 1))
        for c in range(num_classes)
    ]

    logger.info(f"Class counts (train set): {dict(label_counts)}")
    logger.info(f"Class weights applied: {dict(enumerate(round(w, 3) for w in weights))}")

    return torch.tensor(weights, dtype=torch.float32, device=device)


@torch.no_grad()
def evaluate_metrics(model, loader, device):
    """
    Returns (accuracy, macro_f1, balanced_accuracy). Model selection uses
    macro_f1 rather than raw accuracy — see train_semantic.py's
    evaluate_metrics() docstring for the full rationale.
    """
    model.eval()
    all_preds = []
    all_labels = []

    for sequences, labels in loader:
        sequences = sequences.to(device)
        _, logits = model(sequences)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    train_loader, val_loader, _ = get_dataloaders(batch_size=BATCH_SIZE)
    logger.info(
        f"Train samples: {len(train_loader.dataset)} | "
        f"Val samples: {len(val_loader.dataset)}"
    )

    class_weights = compute_class_weights(train_loader.dataset, device=device)

    model = TemporalClassifier().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
    scaler = GradScaler()

    best_val_macro_f1 = -1.0  # ensures epoch 1 always saves a checkpoint, even in a degenerate all-zero-F1 edge case
    epochs_without_improvement = 0

    for epoch in range(EPOCHS):
        model.train()

        total_loss = 0.0
        train_correct = 0
        train_total = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", leave=False)

        for sequences, labels in progress:
            sequences, labels = sequences.to(device), labels.to(device)

            optimizer.zero_grad()

            with autocast():
                _, logits = model(sequences)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

            progress.set_postfix(loss=loss.item())

        train_acc = train_correct / train_total if train_total > 0 else 0.0
        val_acc, val_macro_f1, val_balanced_acc = evaluate_metrics(model, val_loader, device)
        scheduler.step(val_macro_f1)

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch [{epoch + 1}/{EPOCHS}] | "
            f"Train Loss: {total_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val Macro-F1: {val_macro_f1:.4f} | "
            f"Val Balanced Acc: {val_balanced_acc:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        # Model selection uses validation MACRO-F1, not raw accuracy.
        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            epochs_without_improvement = 0

            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch + 1,
                "val_accuracy": val_acc,
                "val_macro_f1": val_macro_f1,
                "val_balanced_accuracy": val_balanced_acc,
            }, CHECKPOINT_PATH)

            logger.info(f"  -> New best checkpoint saved (Val Macro-F1: {val_macro_f1:.4f})")
        else:
            epochs_without_improvement += 1
            logger.info(
                f"  No improvement for {epochs_without_improvement} "
                f"epoch(s) (best: {best_val_macro_f1:.4f})"
            )

        if epochs_without_improvement >= EARLY_STOP_PATIENCE:
            logger.info(
                f"Early stopping triggered after {epoch + 1} epochs "
                f"(no improvement for {EARLY_STOP_PATIENCE} epochs)."
            )
            break

    logger.info("Training complete.")
    logger.info(f"Best Validation Macro-F1: {best_val_macro_f1:.4f}")
    logger.info(f"Checkpoint saved to: {CHECKPOINT_PATH}")
    logger.info(
        "Run `python -m src.modeling.test_temporal` for the final, "
        "one-time test set evaluation."
    )


if __name__ == "__main__":
    main()
