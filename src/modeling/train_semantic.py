import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from src.features.semantic_extractor import SemanticClassifier
from src.preprocessing.image_loader import get_dataloaders

CHECKPOINT_PATH = "saved_models/semantic_checkpoint.pth"


@torch.no_grad()
def evaluate_accuracy(model, loader, device):
    """Quick accuracy-only pass, used for per-epoch validation checks."""
    model.eval()
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        _, logits = model(images)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total if total > 0 else 0.0


def main():
    os.makedirs("saved_models", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train + validation only — the test set is intentionally not loaded
    # here. Model selection must never see it. See test_semantic.py for the
    # one-time final evaluation.
    train_loader, val_loader, _ = get_dataloaders(batch_size=16)

    model = SemanticClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scaler = GradScaler()

    epochs = 5
    best_val_acc = 0.0

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
        val_acc = evaluate_accuracy(model, val_loader, device)

        print(
            f"Epoch [{epoch + 1}/{epochs}] | "
            f"Train Loss: {total_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        # Model selection uses ONLY validation accuracy. The test set is
        # never touched during training.
        if val_acc > best_val_acc:
            best_val_acc = val_acc

            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch + 1,
                "val_accuracy": val_acc,
            }, CHECKPOINT_PATH)

            print(f"  -> New best checkpoint saved (Val Acc: {val_acc:.4f})")

    print("\nTraining complete.")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Checkpoint saved to: {CHECKPOINT_PATH}")
    print("\nRun `python -m src.modeling.test_semantic` for the final, "
          "one-time test set evaluation.")


if __name__ == "__main__":
    main()
