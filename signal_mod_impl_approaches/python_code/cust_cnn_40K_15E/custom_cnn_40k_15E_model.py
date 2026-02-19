"""
custom_cnn_40k_model.py

Custom lightweight convolutional neural network (~39.8 K trainable parameters)
for RF modulation classification from scaleogram images.

Architecture
------------
    Conv2d(3→14, bias=False)   → BN → ReLU → MaxPool     112×112
    Conv2d(14→28, bias=False)  → BN → ReLU → MaxPool      56×56
    Conv2d(28→40, bias=False)  → BN → ReLU → MaxPool      28×28
    Conv2d(40→44, bias=False)  → BN → ReLU → MaxPool      14×14
    AdaptiveAvgPool(3×3)                                    3×3×44 = 396
    Dropout(0.3)
    Linear(396→24) → ReLU → Dropout(0.3)
    Linear(24→9)

Wider channel progression (14→28→40→44) compared to the 30 K model
(10→20→32→36) provides greater feature capacity while staying under
40,000 trainable parameters.  Conv layers use bias=False since
BatchNorm supplies its own learnable bias.

Training is limited to 15 epochs with the StepLR decay adjusted to
step_size=5 (gamma=0.1) so that two learning-rate reductions still
occur within the shortened schedule.

Uses the same dataset produced by build_dataset.py:
    dataset/{train,val,test}/{8psk,16qam,64qam,bfm,bpsk,cpfsk,gfsk,pam4,qpsk}

Performance metric : weighted F1 score (training and validation)
Post-training      : confusion matrices for training and validation sets
"""

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from time import time


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_ROOT = SCRIPT_DIR / "dataset"

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
NUM_CLASSES = 9
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 1e-3
NUM_WORKERS = 0          # set >0 on Linux for speed; 0 is safest on Windows

# ---------------------------------------------------------------------------
# Data transforms  (no ImageNet stats — this is a from-scratch model)
# ---------------------------------------------------------------------------
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

eval_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


# ---------------------------------------------------------------------------
# Custom CNN model  (~39,831 trainable parameters)
# ---------------------------------------------------------------------------
class CustomCNN40K(nn.Module):
    """Lightweight 4-layer CNN (≤40 K params) for 224×224 RGB scaleogram
    classification.  Channel widths are widened relative to CustomCNN30K
    (14→28→40→44 vs 10→20→32→36) and Conv2d bias is disabled (BatchNorm
    supplies its own bias)."""

    def __init__(self, num_classes: int = 9):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 224×224×3 → 112×112×14
            nn.Conv2d(3, 14, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(14),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 112×112×14 → 56×56×28
            nn.Conv2d(14, 28, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(28),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 56×56×28 → 28×28×40
            nn.Conv2d(28, 40, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(40),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4: 28×28×40 → 14×14×44
            nn.Conv2d(40, 44, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(44),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Reduce to fixed 3×3 spatial size → 396-dim vector
            nn.AdaptiveAvgPool2d((3, 3)),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(44 * 3 * 3, 24),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(24, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)      # flatten
        x = self.classifier(x)
        return x


def count_parameters(model):
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Helper — collect all predictions and labels for a dataset
# ---------------------------------------------------------------------------
def collect_predictions(model, dataloader, device):
    """Run the model over *dataloader* and return (all_preds, all_labels)."""
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_model(model, dataloaders, dataset_sizes, criterion, optimizer,
                scheduler, device, num_epochs):
    """Train the model and report weighted F1 each epoch."""

    best_f1 = 0.0
    best_model_weights = model.state_dict()

    train_f1_history = []
    val_f1_history = []
    train_loss_history = []
    val_loss_history = []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 40)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            all_preds = []
            all_labels = []

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)
            epoch_f1 = f1_score(all_labels, all_preds, average="weighted")

            if phase == "train":
                train_loss_history.append(epoch_loss)
                train_f1_history.append(epoch_f1)
            else:
                val_loss_history.append(epoch_loss)
                val_f1_history.append(epoch_f1)

            print(f"  {phase:>5s}  Loss: {epoch_loss:.4f}  "
                  f"Weighted F1: {epoch_f1:.4f}")

            # Keep the best model based on validation F1
            if phase == "val" and epoch_f1 > best_f1:
                best_f1 = epoch_f1
                best_model_weights = model.state_dict().copy()

    print(f"\nBest validation F1: {best_f1:.4f}")
    model.load_state_dict(best_model_weights)

    history = {
        "train_f1": train_f1_history,
        "val_f1": val_f1_history,
        "train_loss": train_loss_history,
        "val_loss": val_loss_history,
    }
    return model, history


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def plot_training_curves(history, save_dir):
    """Save loss and F1 curves to *save_dir*."""
    epochs = range(1, len(history["train_f1"]) + 1)

    # --- Loss ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_loss"], "o-", label="Train Loss")
    ax.plot(epochs, history["val_loss"], "o-", label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Custom CNN 40K — Training and Validation Loss")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(save_dir / "custom_cnn_40k_loss_curves.png", dpi=150)
    plt.close(fig)

    # --- F1 ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_f1"], "o-", label="Train F1")
    ax.plot(epochs, history["val_f1"], "o-", label="Val F1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weighted F1 Score")
    ax.set_title("Custom CNN 40K — Training and Validation F1 Score")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(save_dir / "custom_cnn_40k_f1_curves.png", dpi=150)
    plt.close(fig)
    print(f"Training curves saved to {save_dir}")


def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path):
    """Compute and save a confusion matrix figure."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 9))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")
    ax.set_title(title, fontsize=14)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Datasets and loaders ---
    image_datasets = {
        "train": datasets.ImageFolder(DATASET_ROOT / "train",
                                      transform=train_transforms),
        "val":   datasets.ImageFolder(DATASET_ROOT / "val",
                                      transform=eval_transforms),
    }

    dataloaders = {
        split: DataLoader(ds, batch_size=BATCH_SIZE, shuffle=(split == "train"),
                          num_workers=NUM_WORKERS)
        for split, ds in image_datasets.items()
    }

    dataset_sizes = {split: len(ds) for split, ds in image_datasets.items()}
    class_names = image_datasets["train"].classes  # alphabetical folder names
    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Dataset sizes — train: {dataset_sizes['train']}, "
          f"val: {dataset_sizes['val']}")

    # --- Model ---
    model = CustomCNN40K(num_classes=NUM_CLASSES)
    total_params = count_parameters(model)
    print(f"\nCustom CNN 40K — trainable parameters: {total_params:,}")
    assert total_params <= 40_000, (
        f"Model has {total_params:,} parameters, exceeding the 40,000 limit"
    )
    model = model.to(device)

    # --- Loss, optimiser, scheduler ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # --- Train ---
    model, history = train_model(
        model, dataloaders, dataset_sizes, criterion, optimizer, scheduler,
        device, NUM_EPOCHS,
    )

    # --- Save model ---
    model_path = SCRIPT_DIR / "custom_cnn_40k_rf_mod_classifier.pth"
    torch.save(model.state_dict(), model_path)
    print(f"\nModel weights saved to {model_path}")

    # --- Plot training curves ---
    plot_training_curves(history, SCRIPT_DIR)

    # --- Post-training evaluation: confusion matrices ---
    # Use deterministic (no augmentation) transforms for training set evaluation
    train_eval_dataset = datasets.ImageFolder(DATASET_ROOT / "train",
                                              transform=eval_transforms)
    train_eval_loader = DataLoader(train_eval_dataset, batch_size=BATCH_SIZE,
                                   shuffle=False, num_workers=NUM_WORKERS)

    print("\nGenerating confusion matrices ...")

    # Training set
    train_preds, train_labels = collect_predictions(model, train_eval_loader,
                                                    device)
    train_f1_final = f1_score(train_labels, train_preds, average="weighted")
    print(f"  Train  — Weighted F1: {train_f1_final:.4f}")
    plot_confusion_matrix(
        train_labels, train_preds, class_names,
        f"Custom CNN 40K — Training Set Confusion Matrix  (F1={train_f1_final:.4f})",
        SCRIPT_DIR / "custom_cnn_40k_confusion_matrix_train.png",
    )

    # Validation set
    val_preds, val_labels = collect_predictions(model, dataloaders["val"],
                                                device)
    val_f1_final = f1_score(val_labels, val_preds, average="weighted")
    print(f"  Val    — Weighted F1: {val_f1_final:.4f}")
    plot_confusion_matrix(
        val_labels, val_preds, class_names,
        f"Custom CNN 40K — Validation Set Confusion Matrix  (F1={val_f1_final:.4f})",
        SCRIPT_DIR / "custom_cnn_40k_confusion_matrix_val.png",
    )

    print("\nDone.")


if __name__ == "__main__":
    start_time = time()
    main()
    end_time = time()

    mod_dev_time = end_time - start_time

    print('The model development time is {:2f} seconds over {} epochs'.format(
        mod_dev_time, NUM_EPOCHS))
