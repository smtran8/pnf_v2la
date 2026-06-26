import argparse
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# -----------------------------------------------------------------------
# Hyperparameters
# -----------------------------------------------------------------------
INPUT_SIZE  = 2      # (forward, side) per timestep
D_MODEL     = 64    # embedding dimension — must be divisible by NUM_HEADS => Up this to bring up the parameters, even though prone to overfitting
NUM_HEADS   = 4      # attention heads (d_k = D_MODEL / NUM_HEADS = 16 per head)
NUM_LAYERS  = 2      # number of encoder AND decoder layers
FFN_DIM     = 256    # feed-forward hidden dim (standard: 4 × D_MODEL)
DROPOUT     = 0.1
BATCH_SIZE  = 64
EPOCHS      = 100
LR          = 1e-3
CLIP_GRAD   = 5.0


# -----------------------------------------------------------------------
# Dataset — identical to train_lstm.py
# -----------------------------------------------------------------------
class TrajectoryDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)   # (N, IN_LEN,  2)
        self.y = torch.from_numpy(y)   # (N, OUT_LEN, 2)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -----------------------------------------------------------------------
# Sinusoidal Positional Encoding - For each time step only
# -----------------------------------------------------------------------
class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed (non-learned) positional encoding using sine and cosine waves.

    PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    Each position gets a unique d_model-dimensional fingerprint.
    Same position always gets the same encoding, regardless of input values.
    Adjacent positions have smoothly varying encodings, so the model can
    learn relative position from the difference in PE values.

    The encoding is added (not concatenated) to the input embedding,
    keeping the dimension at d_model.
    """
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute PE matrix once — shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)

        # Frequency scaling: 10000^(2i/d_model) for each dimension pair
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) *
            -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)   # even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)   # odd dimensions

        # Register as buffer: saved with model state but not a learned parameter
        # Shape: (1, max_len, d_model) — the leading 1 allows batch broadcasting
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        Returns x + PE[:seq_len], same shape.
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# -----------------------------------------------------------------------
# Transformer Model
# -----------------------------------------------------------------------
class TrajectoryTransformer(nn.Module):
    """
    Encoder-Decoder Transformer for trajectory forecasting.

    Encoder path:
      input (batch, IN_LEN, 2)
        - linear projection to d_model
        - + sinusoidal positional encoding
        - N × TransformerEncoderLayer (self-attention)
        - memory (batch, IN_LEN, d_model)  ← all positions kept, not compressed

    Decoder path:
      target (batch, OUT_LEN, 2)  [shifted right during training]
        - linear projection to d_model
        - + sinusoidal positional encoding
        - N × TransformerDecoderLayer:
            1. Masked self-attention (can't attend to future output steps)
            2. Cross-attention over encoder memory (the key step —
               each output position directly queries all input positions)
            3. Feed-forward network
        - linear projection to 2
        - predictions (batch, OUT_LEN, 2)
    """
    def __init__(self, input_size=INPUT_SIZE, d_model=D_MODEL, num_heads=NUM_HEADS,
                 num_layers=NUM_LAYERS, ffn_dim=FFN_DIM, dropout=DROPOUT):
        super().__init__()

        # --- Input projections ---
        # Project raw (forward, side) from 2D → d_model dimensions
        # Both encoder and decoder need this since they share the same input format
        self.src_projection = nn.Linear(input_size, d_model)
        self.tgt_projection = nn.Linear(input_size, d_model)

        # --- Positional encoding ---
        # Shared between encoder and decoder — same PE formula for both
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, dropout=dropout)

        # --- Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model    = d_model,
            nhead      = num_heads,
            dim_feedforward = ffn_dim,
            dropout    = dropout,
            batch_first = True    # input shape: (batch, seq_len, d_model)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # --- Decoder ---
        decoder_layer = nn.TransformerDecoderLayer(
            d_model    = d_model,
            nhead      = num_heads,
            dim_feedforward = ffn_dim,
            dropout    = dropout,
            batch_first = True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # --- Output projection ---
        self.output_layer = nn.Linear(d_model, input_size)

        # --- Weight initialization ---
        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialization for linear layers — standard for Transformers."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _make_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        """
        Causal (look-ahead) mask for the decoder's self-attention.
        Prevents position i from attending to positions j > i.
        Shape: (size, size), upper triangle is -inf, diagonal and below is 0.

        Example for size=3:
          [[0,    -inf, -inf],
           [0,    0,    -inf],
           [0,    0,    0   ]]

        When added to attention logits before softmax, -inf → 0 probability.
        """
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))

    def encode(self, src: torch.Tensor) -> torch.Tensor:
        """
        Encode the input sequence.
        src: (batch, IN_LEN, 2)
        Returns memory: (batch, IN_LEN, d_model)

        Unlike LSTM which returns ONE context vector (h, c),
        the Transformer returns ALL IN_LEN encoded positions.
        The decoder can directly attend to any of them via cross-attention.
        """
        src_emb = self.pos_encoding(self.src_projection(src))
        memory  = self.encoder(src_emb)
        return memory

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        Decode given target sequence and encoder memory.
        tgt:    (batch, OUT_LEN, 2)
        memory: (batch, IN_LEN,  d_model)
        Returns: (batch, OUT_LEN, 2)
        """
        tgt_len  = tgt.size(1)
        tgt_mask = self._make_causal_mask(tgt_len, tgt.device)

        tgt_emb = self.pos_encoding(self.tgt_projection(tgt))
        out     = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        return self.output_layer(out)

    def forward(self, src: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        """
        Training forward pass using teacher forcing.

       Re-use Teacher Forcing, but not the probability kind, but the whole sequence.
       Example: Ground Truth: Step 0, 1, 2, 3, ..., 10
       Feed into Decoder: Last Predicted, 0, 1, 2. Decoder 0 receive last predicted and predict step 0, decoder 1 receive step 0 (ground truth) and predict step 1
       Loss is the difference between step 1 prediction and step 1 ground truth. Step 1 prediction is NOT used to feed for the decoder, which is different from LSTM

        At inference: call encode() then decode() autoregressively.

        src: (batch, IN_LEN,  2) — observed trajectory
        y:   (batch, OUT_LEN, 2) — ground truth future (required during training)
        Returns: (batch, OUT_LEN, 2)
        """
        memory = self.encode(src)

        # Shift target right: [last_input, y_0, y_1, ..., y_{N-2}]
        # The decoder predicts y_0 from last_input, y_1 from y_0, etc.
        tgt_input = torch.cat([src[:, -1:, :], y[:, :-1, :]], dim=1)

        return self.decode(tgt_input, memory)

    @torch.no_grad()
    def predict(self, src: torch.Tensor, out_len: int) -> torch.Tensor:
        """
        Autoregressive inference — no ground truth available.
        Generates predictions one step at a time, feeding each back.

        src: (batch, IN_LEN, 2)
        Returns: (batch, out_len, 2)
        """
        memory     = self.encode(src)
        dec_input  = src[:, -1:, :]    # start from last observed position
        predictions = []

        for _ in range(out_len):
            # Decode all positions generated so far
            out = self.decode(dec_input, memory)
            # Take only the last position — the newest prediction
            next_pred = out[:, -1:, :]
            predictions.append(next_pred)
            # Append prediction to decoder input for next step
            dec_input = torch.cat([dec_input, next_pred], dim=1)

        return torch.cat(predictions, dim=1)   # (batch, out_len, 2)


# -----------------------------------------------------------------------
# Metrics — identical to train_lstm.py for direct comparison
# -----------------------------------------------------------------------
def compute_ade_fde(predicted, actual):
    if isinstance(predicted, torch.Tensor):
        predicted = predicted.detach().cpu().numpy()
    if isinstance(actual, torch.Tensor):
        actual = actual.detach().cpu().numpy()

    diff = predicted - actual
    l2   = np.linalg.norm(diff, axis=-1)   # (N, OUT_LEN)

    ade = float(l2.mean())
    fde = float(l2[:, -1].mean())
    return ade, fde


# -----------------------------------------------------------------------
# Training / evaluation loops
# -----------------------------------------------------------------------
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        pred = model(X_batch, y_batch)   # teacher forcing

        loss = criterion(pred, y_batch)
        loss.backward()

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

            # Use autoregressive inference at eval — no teacher forcing
            out_len = y_batch.size(1)
            pred    = model.predict(X_batch, out_len)

            loss = criterion(pred, y_batch)
            total_loss += loss.item() * len(X_batch)

            all_pred.append(pred)
            all_true.append(y_batch)

    all_pred = torch.cat(all_pred, dim=0)
    all_true = torch.cat(all_true, dim=0)
    ade, fde = compute_ade_fde(all_pred, all_true)

    return total_loss / len(loader.dataset), ade, fde


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="P&F Transformer trainer")
    p.add_argument("--data",       required=True,                  help="path to dataset.npz")
    p.add_argument("--out",        default="best_transformer.pt",  help="path to save best model")
    p.add_argument("--d-model",    type=int,   default=D_MODEL)
    p.add_argument("--num-heads",  type=int,   default=NUM_HEADS)
    p.add_argument("--num-layers", type=int,   default=NUM_LAYERS)
    p.add_argument("--ffn-dim",    type=int,   default=FFN_DIM)
    p.add_argument("--dropout",    type=float, default=DROPOUT)
    p.add_argument("--batch-size", type=int,   default=BATCH_SIZE)
    p.add_argument("--epochs",     type=int,   default=EPOCHS)
    p.add_argument("--lr",         type=float, default=LR)
    p.add_argument("--eval-only",  action="store_true",
                   help="skip training, load --out and evaluate on val set")
    args = p.parse_args()

    # Validate head count
    assert args.d_model % args.num_heads == 0, \
        f"d_model ({args.d_model}) must be divisible by num_heads ({args.num_heads})"

    # --- Device ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
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
        batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(
        TrajectoryDataset(X_val, y_val),
        batch_size=args.batch_size, shuffle=False)

    # --- Model ---
    model = TrajectoryTransformer(
        input_size = INPUT_SIZE,
        d_model    = args.d_model,
        num_heads  = args.num_heads,
        num_layers = args.num_layers,
        ffn_dim    = args.ffn_dim,
        dropout    = args.dropout
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")
    print(f"d_model={args.d_model}  heads={args.num_heads}  "
          f"d_k={args.d_model // args.num_heads}  layers={args.num_layers}  "
          f"ffn_dim={args.ffn_dim}")

    # --- Eval only ---
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10)

    best_val_ade = float("inf")
    print(f"\n{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>10} {'Val ADE':>10} {'Val FDE':>10}")
    print("─" * 56)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_ade, val_fde = eval_epoch(model, val_loader, criterion, device)

        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']

        print(f"{epoch:>6}  {train_loss:>12.6f}  {val_loss:>10.6f}  "
              f"{val_ade:>10.4f}m  {val_fde:>10.4f}m")

        if new_lr != old_lr:
            print(f"LR reduced: {old_lr:.2e} → {new_lr:.2e}")

        if val_ade < best_val_ade:
            best_val_ade = val_ade
            torch.save(model.state_dict(), args.out)
            print(f"best model saved (ADE={best_val_ade:.4f}m)")

    print(f"\nTraining complete. Best val ADE: {best_val_ade:.4f}m")
    print(f"Model saved to: {args.out}")
    print(f"\nComparison:")
    print(f"  Constant-velocity  ADE=0.7842m  FDE=1.4009m")
    print(f"  LSTM               ADE=0.1659m  FDE=0.2380m")
    print(f"  Transformer        ADE={best_val_ade:.4f}m  FDE=?  (run --eval-only for FDE)")


if __name__ == "__main__":
    main()