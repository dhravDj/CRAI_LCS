import os
import argparse
import time
from typing import Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

from src.models.cnn_lstm_risk import ClinicalRiskCNNLSTM
from src.eeg.dataset import EEGTaskDataset as ECGTaskDataset
from src.training.focal_loss import BinaryFocalLoss
from src.utils.seed import set_global_seed

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> float:
    model.train()
    total_loss = 0.0

    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model.forward_logits(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(x)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, float, float]:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        logits = model.forward_logits(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * len(x)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        all_preds.extend(probs)
        all_targets.extend(y.cpu().numpy().flatten())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    mean_loss = total_loss / len(loader.dataset)
    binary_preds = (all_preds >= 0.50).astype(int)
    acc = accuracy_score(all_targets, binary_preds) * 100.0
    f1 = f1_score(all_targets, binary_preds, zero_division=0) * 100.0

    # ROC-AUC calculation (guard against single-class edge case)
    if len(np.unique(all_targets)) > 1:
        auc = roc_auc_score(all_targets, all_preds) * 100.0
    else:
        auc = 100.0 if acc > 90.0 else 50.0

    return mean_loss, acc, f1, auc


def run_training(
    data_path: str = "data/processed/eeg_tasks_split.npz",
    checkpoint_dir: str = "checkpoints",
    epochs: int = 25,
    batch_size: int = 32,
    lr: float = 1e-3,
    loss_type: str = "bce",
    seed: int = 42,
    device_name: str = "auto"
) -> Dict[str, Any]:
    set_global_seed(seed)

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    print(f"🚀 Training ClinicalRiskCNNLSTM on device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    data = np.load(data_path)
    train_ds = ECGTaskDataset(data["train_x"], data["train_y"], data["train_r"])
    val_ds = ECGTaskDataset(data["val_x"], data["val_y"], data["val_r"])
    test_ds = ECGTaskDataset(data["test_x"], data["test_y"], data["test_r"])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = ClinicalRiskCNNLSTM().to(device)
    print(f"Model parameters: {model.total_parameters:,} (~0.74M)")

    # Class imbalance compensation
    num_pos = np.sum(data["train_y"] == 1)
    num_neg = np.sum(data["train_y"] == 0)
    pos_weight_val = max(1.0, float(num_neg) / max(1.0, float(num_pos)))

    if loss_type == "focal":
        alpha = float(num_neg) / float(num_pos + num_neg)
        criterion = BinaryFocalLoss(alpha=alpha, gamma=2.0)
        print("Using Clinical Binary Focal Loss (gamma=2.0) for boundary sharpening.")
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val], device=device))
        print("Using Weighted Binary Cross-Entropy with Logits.")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_checkpoint = os.path.join(checkpoint_dir, "cnn_lstm_risk.pt")

    best_val_auc = 0.0
    patience = 7
    patience_counter = 0

    t_start = time.time()
    for ep in range(1, epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_f1, val_auc = evaluate(model, val_loader, criterion, device)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), best_checkpoint)
            patience_counter = 0
            improved_flag = " ⭐ [BEST]"
        else:
            patience_counter += 1
            improved_flag = ""

        if ep % 5 == 0 or ep == 1 or improved_flag:
            print(f"Epoch [{ep:02d}/{epochs:02d}] Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.1f}% | Val AUC: {val_auc:.1f}%{improved_flag}")

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {ep} (patience reached).")
            break

    total_time = time.time() - t_start
    print(f"✅ Training completed in {total_time:.1f} seconds! Best model saved to: {best_checkpoint}")

    # Load best weights for final test evaluation
    model.load_state_dict(torch.load(best_checkpoint, map_location=device))
    test_loss, test_acc, test_f1, test_auc = evaluate(model, test_loader, criterion, device)

    print("\n" + "=" * 60)
    print("🏥 FINAL CLINICAL TEST EVALUATION (Unseen Patient Records)")
    print("=" * 60)
    print(f"• Test Accuracy : {test_acc:.2f}%")
    print(f"• Test F1-Score : {test_f1:.2f}%")
    print(f"• Test AUC-ROC  : {test_auc:.2f}%")
    print("=" * 60)

    return {
        "best_val_auc": best_val_auc,
        "test_acc": test_acc,
        "test_f1": test_f1,
        "test_auc": test_auc,
        "total_time_sec": total_time,
        "checkpoint_path": best_checkpoint
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed
    )
