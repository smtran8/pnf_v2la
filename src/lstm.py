#LSTM = Long-Short Term Memory
#Hidden State = Short Term
#Cell State = Long Term

#Forget Gate: How much information to keep in cell state; Input Gate: How much information to write to cell state; Output Gate: How much of cell state to use for hidden state
"""
train_lstm.py — P&F LSTM Encoder-Decoder Forecaster

Architecture: sequence-to-sequence encoder-decoder
  - Encoder: LSTM reads 25 observed positions → context vector (h, c)
  - Decoder: LSTM generates 25 predicted positions autoregressively
  - Output:  linear layer maps hidden state → (forward, side) per step

Input:  dataset.npz produced by make_dataset.py
Output: best_lstm.pt  — saved model weights (best validation ADE)

Evaluate (prints ADE/FDE against the val set):
    python train_lstm.py --data dataset.npz --out best_lstm.pt --eval-only
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# Hyperparameters 

INPUT_SIZE   = 2        # (forward, side) per timestep
HIDDEN_SIZE  = 128      # size of LSTM hidden/cell state
NUM_LAYERS   = 2        # stacked LSTM layers (depth)
DROPOUT      = 0.1      # dropout between LSTM layers (only active if NUM_LAYERS > 1)
BATCH_SIZE   = 64
EPOCHS       = 100
LR           = 1e-3     # Adam learning rate
CLIP_GRAD    = 5.0      # gradient clipping — prevents exploding gradients
TEACHER_FORCE_RATIO = 0.5   # probability of using ground-truth input in decoder


# Dataset
class TrajectoryDataset(Dataset):
    """
    Wraps the X (input) and y (target) arrays from dataset.npz.
    Each item is one (input_window, target_window) pair.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        # X: (N, IN_LEN,  2)  float32
        # y: (N, OUT_LEN, 2)  float32
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]



# Model

class Encoder(nn.Module):
    """
    Reads the full input sequence and compresses it into (h, c).
    h and c are each shape (NUM_LAYERS, batch, HIDDEN_SIZE).
    """
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = dropout if num_layers > 1 else 0.0,
            batch_first = True   # input shape: (batch, seq_len, input_size)
        )

    def forward(self, x):
        # x: (batch, IN_LEN, 2)
        _, (h, c) = self.lstm(x)
        # h: (NUM_LAYERS, batch, HIDDEN_SIZE) — short-term context
        # c: (NUM_LAYERS, batch, HIDDEN_SIZE) — long-term context
        return h, c


class Decoder(nn.Module):
    """
    Generates one predicted position at a time, autoregressively.
    Initialized from the encoder's (h, c) context vectors.
    """
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = dropout if num_layers > 1 else 0.0,
            batch_first = True
        )
        # Maps hidden state → (forward, side) prediction
        self.output_layer = nn.Linear(hidden_size, input_size)

    def forward_step(self, x, h, c):
        """
        One decoder step: takes current input + (h, c), returns prediction + new (h, c).
        x: (batch, 1, 2)  — one timestep
        """
        out, (h, c) = self.lstm(x, (h, c))
        # out: (batch, 1, HIDDEN_SIZE)
        pred = self.output_layer(out)
        # pred: (batch, 1, 2)
        return pred, h, c


class EncoderDecoder(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE,
                 num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.encoder = Encoder(input_size, hidden_size, num_layers, dropout)
        self.decoder = Decoder(input_size, hidden_size, num_layers, dropout)

    def forward(self, x, y=None, teacher_force_ratio=0.0):
        """
        Args:
            x                  : (batch, IN_LEN, 2)  — observed trajectory
            y                  : (batch, OUT_LEN, 2) — ground truth future (for teacher forcing)
            teacher_force_ratio: probability of feeding ground truth to decoder instead
                                 of its own prediction. Set to 0 at eval time.

        Returns:
            predictions: (batch, OUT_LEN, 2)
        """
        batch_size = x.size(0)
        out_len    = y.size(1) if y is not None else 15

        # --- Encode ---
        h, c = self.encoder(x)

        # --- Decode autoregressively ---
        # First decoder input is the last observed position
        dec_input = x[:, -1:, :]   # (batch, 1, 2)

        predictions = []
        for t in range(out_len):
            pred, h, c = self.decoder.forward_step(dec_input, h, c)
            predictions.append(pred)

            # Teacher forcing: randomly feed ground truth instead of prediction
            # This stabilizes early training when predictions are poor.
            # At eval time, teacher_force_ratio=0 so we always feed our own prediction.
            if y is not None and torch.rand(1).item() < teacher_force_ratio:
                dec_input = y[:, t:t+1, :]   # ground truth for next step
            else:
                dec_input = pred              # model's own prediction

        return torch.cat(predictions, dim=1)  # (batch, OUT_LEN, 2)


#Metrics
def compute_ade_fde(predicted, actual):
    """
    predicted, actual: (N, OUT_LEN, 2)  tensors or numpy arrays
    Returns: ADE (scalar), FDE (scalar), both in meters
    """
    if isinstance(predicted, torch.Tensor):
        predicted = predicted.detach().cpu().numpy()
    if isinstance(actual, torch.Tensor):
        actual = actual.detach().cpu().numpy()

    # L2 distance at each step: (N, OUT_LEN)
    diff   = predicted - actual
    l2     = np.linalg.norm(diff, axis=-1)

    ade = float(l2.mean())               # mean over all steps and all windows
    fde = float(l2[:, -1].mean())        # mean over final step only
    return ade, fde


# Training loop

def train_epoch(model, loader, optimizer, criterion, device, teacher_force_ratio):
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        pred = model(X_batch, y_batch, teacher_force_ratio=teacher_force_ratio)

        loss = criterion(pred, y_batch)
        loss.backward()
        #Let's see if we do have any exploding gradient
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
        # clip_grad_norm_ returns the norm BEFORE clipping
        if grad_norm > CLIP_GRAD:
            print(f"Gradient clipped: from {grad_norm:.2f} to {CLIP_GRAD}")

        # Gradient clipping — prevents exploding gradients common in RNNs
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)

        optimizer.step()
        total_loss += loss.item() * len(X_batch)

    return total_loss / len(loader.dataset)


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_pred   = []
    all_true   = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # No teacher forcing at eval time
            pred = model(X_batch, y_batch, teacher_force_ratio=0.0)

            loss = criterion(pred, y_batch)
            total_loss += loss.item() * len(X_batch)

            all_pred.append(pred)
            all_true.append(y_batch)

    all_pred = torch.cat(all_pred, dim=0)
    all_true = torch.cat(all_true, dim=0)
    ade, fde = compute_ade_fde(all_pred, all_true)

    return total_loss / len(loader.dataset), ade, fde


# Main
def main():
    p = argparse.ArgumentParser(description="P&F LSTM encoder-decoder trainer")
    p.add_argument("--data",       required=True,           help="path to dataset.npz")
    p.add_argument("--out",        default="best_lstm.pt",  help="path to save best model")
    p.add_argument("--hidden",     type=int,   default=HIDDEN_SIZE)
    p.add_argument("--layers",     type=int,   default=NUM_LAYERS)
    p.add_argument("--dropout",    type=float, default=DROPOUT)
    p.add_argument("--batch-size", type=int,   default=BATCH_SIZE)
    p.add_argument("--epochs",     type=int,   default=EPOCHS)
    p.add_argument("--lr",         type=float, default=LR)
    p.add_argument("--tf-ratio",   type=float, default=TEACHER_FORCE_RATIO,
                   help="teacher forcing ratio (0=never, 1=always)")
    p.add_argument("--eval-only",  action="store_true",
                   help="skip training, load --out and evaluate on val set")
    args = p.parse_args()

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load dataset ---
    data    = np.load(args.data)
    X_train = data["X_train"].astype(np.float32)
    y_train = data["y_train"].astype(np.float32)
    X_val   = data["X_val"].astype(np.float32)
    y_val   = data["y_val"].astype(np.float32)

    print(f"Train: {X_train.shape[0]} windows  |  Val: {X_val.shape[0]} windows")
    print(f"Input shape: {X_train.shape[1:]}  |  Target shape: {y_train.shape[1:]}")

    train_loader = DataLoader(
        TrajectoryDataset(X_train, y_train),
        batch_size=args.batch_size, shuffle=True,  drop_last=True)
    val_loader = DataLoader(
        TrajectoryDataset(X_val, y_val),
        batch_size=args.batch_size, shuffle=False)

    # --- Model ---
    model = EncoderDecoder(
        input_size  = INPUT_SIZE,
        hidden_size = args.hidden,
        num_layers  = args.layers,
        dropout     = args.dropout
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    # --- Eval only mode ---
    if args.eval_only:
        model.load_state_dict(torch.load(args.out, map_location=device))
        criterion = nn.MSELoss()
        _, ade, fde = eval_epoch(model, val_loader, criterion, device)
        print(f"\nEval results on val set:")
        print(f"  ADE = {ade:.4f}m")
        print(f"  FDE = {fde:.4f}m")
        return

    # --- Training ---
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # Learning rate scheduler: halve LR if val loss doesn't improve for 10 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10)

    best_val_ade = float("inf")
    print(f"\n{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>10} {'Val ADE':>10} {'Val FDE':>10}")
    print("─" * 56)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, args.tf_ratio)
        val_loss, val_ade, val_fde = eval_epoch(
            model, val_loader, criterion, device)

#Check if Learning Rate change:
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != old_lr:
            print(f"  LR reduced: from {old_lr:.2e} to {new_lr:.2e}")

        print(f"{epoch:>6}  {train_loss:>12.6f}  {val_loss:>10.6f}  "
              f"{val_ade:>10.4f}m  {val_fde:>10.4f}m")

        # Save best model by val ADE (not val loss — ADE is what we compare to baseline)
        if val_ade < best_val_ade:
            best_val_ade = val_ade
            torch.save(model.state_dict(), args.out)
            print(f"Best model saved (ADE={best_val_ade:.4f}m)")

    print(f"\nTraining complete. Best val ADE: {best_val_ade:.4f}m")
    print(f"Model saved to: {args.out}")
    print(f"\nConstant-velocity baseline for comparison:")
    print(f"  ADE = 0.7842m  |  FDE = 1.4009m")


if __name__ == "__main__":
    main()