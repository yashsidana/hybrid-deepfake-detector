import logging
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

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


@torch.no_grad()
def evaluate_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    for sequences, labels in loader:
        sequences, labels = sequences.to(device), labels.to(device)
        _, logits = model(sequences)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total if total > 0 else 0.0


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    train_loader, val_loader, _ = get_dataloaders(batch_size=BATCH_SIZE)
    logger.info(
        f"Train samples: {len(train_loader.dataset)} | "
        f"Val samples: {len(val_loader.dataset)}"
    )

    model = TemporalClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
    scaler = GradScaler()

    best_val_acc = 0.0
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
        val_acc = evaluate_accuracy(model, val_loader, device)
        scheduler.step(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch [{epoch + 1}/{EPOCHS}] | "
            f"Train Loss: {total_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0

            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch + 1,
                "val_accuracy": val_acc,
            }, CHECKPOINT_PATH)

            logger.info(f"  -> New best checkpoint saved (Val Acc: {val_acc:.4f})")
        else:
            epochs_without_improvement += 1
            logger.info(
                f"  No improvement for {epochs_without_improvement} "
                f"epoch(s) (best: {best_val_acc:.4f})"
            )

        if epochs_without_improvement >= EARLY_STOP_PATIENCE:
            logger.info(
                f"Early stopping triggered after {epoch + 1} epochs "
                f"(no improvement for {EARLY_STOP_PATIENCE} epochs)."
            )
            break

    logger.info("Training complete.")
    logger.info(f"Best Validation Accuracy: {best_val_acc:.4f}")
    logger.info(f"Checkpoint saved to: {CHECKPOINT_PATH}")
    logger.info(
        "Run `python -m src.modeling.test_temporal` for the final, "
        "one-time test set evaluation."
    )


if __name__ == "__main__":
    main()
