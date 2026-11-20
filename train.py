#!/usr/bin/env python
"""
Model Training and MLflow Experiment Tracking for Bangla Handwritten Character Recognition.
Meets Assignment 04 specifications:
- Trains deep CNN on BanglaLekha-Isolated dataset
- Logs experiments, hyperparameters, per-epoch metrics, and model artifacts to MLflow
- Restores best validation accuracy weights before saving to models/model.keras
- Fixes double-softmax gradient collapse via NLLLoss on log-probabilities
"""

import os
import sys
import copy
import argparse
from pathlib import Path
from typing import Optional

# Enforce PyTorch backend for Keras 3 and allow local filesystem store for MLflow
os.environ["KERAS_BACKEND"] = "torch"
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import keras
import mlflow

from src.dataset import create_dataloaders, find_dataset_location
from src.model import build_model


def get_tracking_uri() -> str:
    """Return SQLite URI for artifacts/mlflow/mlflow.db."""
    mlflow_dir = Path("artifacts") / "mlflow"
    mlflow_dir.mkdir(parents=True, exist_ok=True)
    db_file = (mlflow_dir / "mlflow.db").resolve().as_posix()
    return f"sqlite:///{db_file}"


def compute_top_k_acc(outputs: torch.Tensor, targets: torch.Tensor, k: int = 3) -> float:
    """Compute top-k classification accuracy."""
    with torch.no_grad():
        _, pred = outputs.topk(k, dim=1, largest=True, sorted=True)
        correct = pred.eq(targets.view(-1, 1).expand_as(pred))
        return correct[:, :k].reshape(-1).float().sum().item() / targets.size(0)


def train_single_run(
    run_name: str,
    epochs: int = 15,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    optimizer_type: str = "adam",
    weight_decay: float = 1e-4,
    samples_per_class: Optional[int] = 300,
    dataset_path: Optional[str] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> dict:
    """Execute a single training run logged to MLflow."""
    print(f"\n{'='*70}")
    print(f" Starting MLflow Training Run: {run_name}")
    print(f" Device: {device} | Optimizer: {optimizer_type} | LR: {learning_rate} | Epochs: {epochs}")
    print(f"{'='*70}")

    train_loader, val_loader = create_dataloaders(
        dataset_path=dataset_path,
        batch_size=batch_size,
        val_split=0.15,
        max_samples_per_class=samples_per_class,
        target_size=(64, 64)
    )

    keras_model = build_model(input_shape=(64, 64, 1), num_classes=84)
    keras_model.to(device)

    # Use NLLLoss with log of clamped softmax probabilities to eliminate double-softmax collapse
    criterion = nn.NLLLoss()

    if optimizer_type.lower() == "adamw":
        optimizer = optim.AdamW(keras_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(keras_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_weights = None
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_top3_acc": []}

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "run_name": run_name,
            "architecture": "BanglaOCR_CNN_4Block",
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "optimizer": optimizer_type,
            "weight_decay": weight_decay,
            "samples_per_class": samples_per_class if samples_per_class else "all",
            "input_resolution": "64x64",
            "total_classes": 84,
            "device": device
        })

        for epoch in range(1, epochs + 1):
            # Training phase
            keras_model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0

            for images, labels in train_loader:
                # Keras model expects (N, H, W, C) in channel-last format
                images_channels_last = images.permute(0, 2, 3, 1).to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = keras_model(images_channels_last, training=True)
                log_probs = torch.log(torch.clamp(outputs, min=1e-7, max=1.0))
                loss = criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * labels.size(0)
                _, preds = outputs.max(1)
                correct_train += preds.eq(labels).sum().item()
                total_train += labels.size(0)

            scheduler.step()

            train_loss = running_loss / max(total_train, 1)
            train_acc = correct_train / max(total_train, 1)

            # Validation phase
            keras_model.eval()
            val_running_loss = 0.0
            val_correct = 0
            val_top3_sum = 0.0
            total_val = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images_channels_last = images.permute(0, 2, 3, 1).to(device)
                    labels = labels.to(device)

                    outputs = keras_model(images_channels_last, training=False)
                    log_probs = torch.log(torch.clamp(outputs, min=1e-7, max=1.0))
                    loss = criterion(log_probs, labels)

                    val_running_loss += loss.item() * labels.size(0)
                    _, preds = outputs.max(1)
                    val_correct += preds.eq(labels).sum().item()
                    val_top3_sum += compute_top_k_acc(outputs, labels, k=3) * labels.size(0)
                    total_val += labels.size(0)

            val_loss = val_running_loss / max(total_val, 1)
            val_acc = val_correct / max(total_val, 1)
            val_top3_acc = val_top3_sum / max(total_val, 1)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_top3_acc"].append(val_top3_acc)

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_top3_acc": val_top3_acc
            }, step=epoch)

            print(f"Epoch [{epoch:02d}/{epochs:02d}] "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val Top-3: {val_top3_acc*100:.2f}%")

            # Checkpoint best weights in-memory
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_weights = [w.copy() for w in keras_model.get_weights()]

        mlflow.log_metric("best_val_acc", best_val_acc)

    # Restore best checkpoint weights before returning
    if best_weights is not None:
        keras_model.set_weights(best_weights)
        print(f"Restored best weights with Val Acc = {best_val_acc*100:.2f}%")

    return {
        "run_name": run_name,
        "best_val_acc": best_val_acc,
        "final_val_acc": val_acc,
        "keras_model": keras_model
    }


def main():
    parser = argparse.ArgumentParser(description="Train Bangla OCR and Track with MLflow")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs per run")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--samples-per-class", type=int, default=300, help="Max samples per class (None for full)")
    parser.add_argument("--single-run", action="store_true", help="Execute single run only")
    parser.add_argument("--dataset-path", type=str, default=None, help="Explicit path to dataset folder or zip")
    args = parser.parse_args()

    # Configure MLflow local tracking
    tracking_uri = get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = "bangla-handwritten-recognition"
    mlflow.set_experiment(experiment_name)
    print(f"MLflow Tracking URI: {tracking_uri}")
    print(f"MLflow Experiment: {experiment_name}")

    os.makedirs("models", exist_ok=True)
    checkpoint_path = os.path.join("models", "model.keras")

    if args.single_run:
        res = train_single_run(
            run_name="Baseline_Adam",
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            optimizer_type="adam",
            samples_per_class=args.samples_per_class,
            dataset_path=args.dataset_path
        )
        print(f"\nSaving best model checkpoint to {checkpoint_path}...")
        res["keras_model"].save(checkpoint_path)
    else:
        # Assignment Section 2: 'At least two MLflow training runs'
        print("\nExecuting 2 comparative MLflow training runs as required by assignment...")

        # Run 1: Baseline Adam optimizer
        res1 = train_single_run(
            run_name="Run_1_Baseline_Adam",
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            optimizer_type="adam",
            weight_decay=0.0,
            samples_per_class=args.samples_per_class,
            dataset_path=args.dataset_path
        )

        # Run 2: Tuned AdamW with weight decay
        res2 = train_single_run(
            run_name="Run_2_Tuned_AdamW_Reg",
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr * 0.5,
            optimizer_type="adamw",
            weight_decay=1e-4,
            samples_per_class=args.samples_per_class,
            dataset_path=args.dataset_path
        )

        print("\n" + "=" * 70)
        print(" MLflow Runs Comparison:")
        print(f" Run 1 (Baseline Adam): Best Val Acc = {res1['best_val_acc']*100:.2f}%")
        print(f" Run 2 (Tuned AdamW):  Best Val Acc = {res2['best_val_acc']*100:.2f}%")
        print("=" * 70)

        best_res = res2 if res2["best_val_acc"] >= res1["best_val_acc"] else res1
        print(f"Selected superior model from {best_res['run_name']}. Saving to {checkpoint_path}...")
        best_res["keras_model"].save(checkpoint_path)

    print(f"\n[DONE] Training finished successfully. Checkpoint saved: {checkpoint_path}")


if __name__ == "__main__":
    main()
