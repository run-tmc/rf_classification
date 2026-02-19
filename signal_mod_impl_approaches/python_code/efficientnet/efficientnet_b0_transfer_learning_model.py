"""
efficientnet_b0_transfer_learning_model.py

Transfer learning with EfficientNet-B0 for RF modulation classification
from scaleogram images.  Uses the dataset produced by build_dataset.py:
    dataset/{train,val,test}/{8psk,16qam,64qam,bfm,bpsk,cpfsk,gfsk,pam4,qpsk}

Performance metric : weighted F1 score (training and validation)
Post-training      : confusion matrices for training and validation sets

Architecture notes
------------------
EfficientNet-B0 uses compound scaling (depth, width, resolution) to achieve
strong accuracy with fewer parameters than comparable models.  Its classifier
head is a two-element Sequential:
    Dropout(p=0.2) -> Linear(1280, 1000)
For transfer learning we replace only the final Linear layer (classifier[1])
so that the backbone feature extractor (1280-dim output) is preserved.
"""

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
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
NUM_EPOCHS = 25
LEARNING_RATE = 1e-4
NUM_WORKERS = 0          # set >0 on Linux for speed; 0 is safest on Windows

# ---------------------------------------------------------------------------
# Data transforms
# ---------------------------------------------------------------------------
# ImageNet normalisation values
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ---------------------------------------------------------------------------
# Helper -- collect all predictions and labels for a dataset
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
    ax.set_title("EfficientNet-B0 — Training and Validation Loss")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(save_dir / "efficientnet_b0_loss_curves.png", dpi=150)
    plt.close(fig)

    # --- F1 ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_f1"], "o-", label="Train F1")
    ax.plot(epochs, history["val_f1"], "o-", label="Val F1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weighted F1 Score")
    ax.set_title("EfficientNet-B0 — Training and Validation F1 Score")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(save_dir / "efficientnet_b0_f1_curves.png", dpi=150)
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

    # --- Model: EfficientNet-B0 with modified classifier head ---
    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
    )

    # The default classifier is:
    #   Sequential(
    #     [0] Dropout(p=0.2, inplace=True)
    #     [1] Linear(1280, 1000)   <-- replace this layer
    #   )
    in_features = model.classifier[1].in_features   # 1280
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
    model = model.to(device)

    print(f"\nEfficientNet-B0 classifier head:")
    print(f"  {model.classifier}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters()
                           if p.requires_grad)
    print(f"  Total parameters     : {total_params:,}")
    print(f"  Trainable parameters : {trainable_params:,}")

    # --- Loss, optimiser, scheduler ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # --- Train ---
    model, history = train_model(
        model, dataloaders, dataset_sizes, criterion, optimizer, scheduler,
        device, NUM_EPOCHS,
    )

    # --- Save model ---
    model_path = SCRIPT_DIR / "efficientnet_b0_rf_mod_classifier.pth"
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
        f"EfficientNet-B0 Training Set Confusion Matrix  "
        f"(F1={train_f1_final:.4f})",
        SCRIPT_DIR / "efficientnet_b0_confusion_matrix_train.png",
    )

    # Validation set
    val_preds, val_labels = collect_predictions(model, dataloaders["val"],
                                                device)
    val_f1_final = f1_score(val_labels, val_preds, average="weighted")
    print(f"  Val    — Weighted F1: {val_f1_final:.4f}")
    plot_confusion_matrix(
        val_labels, val_preds, class_names,
        f"EfficientNet-B0 Validation Set Confusion Matrix  "
        f"(F1={val_f1_final:.4f})",
        SCRIPT_DIR / "efficientnet_b0_confusion_matrix_val.png",
    )

    print("\nDone.")


if __name__ == "__main__":
    start_time = time()
    main()
    end_time = time()

    mod_dev_time = end_time - start_time 

    print('The model development time is {:2f} seconds over {} epochs'.format(mod_dev_time, NUM_EPOCHS))