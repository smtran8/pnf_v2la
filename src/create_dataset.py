import argparse
import json
import os
import glob
import numpy as np


#Will tune these if needed
IN_LEN    = 25    # number of observed timesteps fed as model input
OUT_LEN   = 25    # number of future timesteps the model must predict
STEP      = 1     # sliding window stride (1 = maximum overlap, fine for training)


# At 25Hz, nominal frame interval is ~0.04s, let's use 0.2 for now
MAX_GAP_S = 0.2  # seconds


def load_session(path: str) -> dict:
    """
    Load one session JSON file.
    Returns: {track_id (int): [(timestamp, forward, side), ...]}
    JSON keys are strings (JSON limitation), converted to ints here.
    """
    with open(path) as f:
        raw = json.load(f)
    return {int(tid): [tuple(pt) for pt in readings]
            for tid, readings in raw.items()}


def make_windows(trajectory: list, in_len: int, out_len: int,
                 step: int, max_gap_s: float):
    """
    Slide an (in_len + out_len) window along one track's trajectory and
    return valid (input, target) pairs.

    Args:
        trajectory : list of (timestamp, forward, side) — one track, one session.
        in_len     : number of steps for the model input (observed history).
        out_len    : number of steps for the model target (future to predict).
        step       : stride between window start positions.
        max_gap_s  : maximum allowed timestamp gap inside any window.

    Returns:
        inputs  : list of np.ndarray shape (in_len,  2)  [forward, side]
        targets : list of np.ndarray shape (out_len, 2)  [forward, side]
    """
    total = in_len + out_len
    n = len(trajectory)

    if n < total:
        return [], []   # trajectory too short to form even one window

    inputs, targets = [], []

    for start in range(0, n - total + 1, step):
        chunk = trajectory[start : start + total]

        # --- Gap check: discard any window containing a timestamp jump ---
        # This catches sensor dropouts, occlusion recovery, and session
        # boundary splices (though those should never happen if we process
        # one session at a time, it's a good safety net either way).
        timestamps = [pt[0] for pt in chunk]
        gaps = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        if max(gaps) > max_gap_s:
            continue  # real discontinuity inside this window — skip it

        # --- Extract positions (drop timestamps) ---
        positions = np.array([[pt[1], pt[2]] for pt in chunk], dtype=np.float32)
        inputs.append(positions[:in_len])
        targets.append(positions[in_len:])

    return inputs, targets


def process_sessions(sessions_dir: str, in_len: int, out_len: int,
                     step: int, max_gap_s: float):
    """
    Loop over every JSON file in sessions_dir, extract all (input, target)
    window pairs across all tracks in each session, and return them pooled.
    """
    json_files = sorted(glob.glob(os.path.join(sessions_dir, "*.json")))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in: {sessions_dir}")

    all_inputs, all_targets = [], []
    total_windows = 0
    total_skipped_gap = 0

    print(f"Found {len(json_files)} session file(s).")
    print(f"Window: {in_len}-in / {out_len}-out, stride={step}, max_gap={max_gap_s}s\n")

    for path in json_files:
        session_name = os.path.basename(path)
        session_data = load_session(path)

        session_windows = 0
        for tid, trajectory in session_data.items():
            before = len(all_inputs)
            inp, tgt = make_windows(trajectory, in_len, out_len, step, max_gap_s)
            all_inputs.extend(inp)
            all_targets.extend(tgt)
            added = len(inp)
            session_windows += added

        total_windows += session_windows
        print(f"  {session_name}: "
              f"{sum(len(v) for v in session_data.values())} total points, "
              f"{len(session_data)} track(s), "
              f"{session_windows} window(s) produced")

    print(f"\nTotal windows: {total_windows}")
    return all_inputs, all_targets


def train_val_split(inputs: list, targets: list, val_ratio=0.15):
    """
    Split the pooled windows into train and validation sets.

    Shuffle globally before splitting (not by session), which is
    acceptable because each window is already a self-contained (input, target)
    pair and we have already prevented cross-session contamination upstream.
    The shuffle ensures train and val get a representative mix of maneuvers.
    """
    n = len(inputs)
    indices = np.random.permutation(n)
    val_size = int(n * val_ratio)

    val_idx   = indices[:val_size]
    train_idx = indices[val_size:]

    inputs_arr  = np.stack(inputs,  axis=0)   # (N, IN_LEN,  2)
    targets_arr = np.stack(targets, axis=0)   # (N, OUT_LEN, 2)

    return (inputs_arr[train_idx], targets_arr[train_idx],
            inputs_arr[val_idx],   targets_arr[val_idx])


def main():
    p = argparse.ArgumentParser(description="P&F Job B: build windowed dataset")
    p.add_argument("--sessions", required=True,
                   help="folder containing session_*.json files from Job A")
    p.add_argument("--out", default="dataset.npz",
                   help="output .npz file path (default: dataset.npz)")
    p.add_argument("--in-len",  type=int, default=IN_LEN,
                   help=f"input window length (default: {IN_LEN})")
    p.add_argument("--out-len", type=int, default=OUT_LEN,
                   help=f"output window length (default: {OUT_LEN})")
    p.add_argument("--step",    type=int, default=STEP,
                   help=f"sliding window stride (default: {STEP})")
    p.add_argument("--max-gap", type=float, default=MAX_GAP_S,
                   help=f"max timestamp gap (s) inside a window (default: {MAX_GAP_S})")
    p.add_argument("--val-ratio", type=float, default=0.15,
                   help="fraction of windows held out for validation (default: 0.15)")
    p.add_argument("--seed", type=int, default=42,
                   help="random seed for reproducible train/val split")
    args = p.parse_args()

    np.random.seed(args.seed)

    # --- Process all sessions ---
    all_inputs, all_targets = process_sessions(
        args.sessions, args.in_len, args.out_len, args.step, args.max_gap)

    if not all_inputs:
        print("ERROR: no valid windows produced. Check your session files "
              "and window configuration (is IN_LEN + OUT_LEN > trajectory length?)")
        return

    # --- Train / val split ---
    X_train, y_train, X_val, y_val = train_val_split(
        all_inputs, all_targets, val_ratio=args.val_ratio)

    # --- Save ---
    np.savez_compressed(args.out,
                        X_train=X_train, y_train=y_train,
                        X_val=X_val,     y_val=y_val)

    print(f"\nDataset saved to: {args.out}")
    print(f"  X_train: {X_train.shape}   y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape}     y_val:   {y_val.shape}")
    print(f"\nInput shape per window:  ({args.in_len},  2)  [forward, side]")
    print(f"Target shape per window: ({args.out_len}, 2)  [forward, side]")
    # print("\nLoad in PyTorch with:")
    # print("  data = np.load('dataset.npz')")
    # print("  X_train = torch.from_numpy(data['X_train'])  # (N, IN_LEN, 2)")
    # print("  y_train = torch.from_numpy(data['y_train'])  # (N, OUT_LEN, 2)")


if __name__ == "__main__":
    main()
    
#To run:
#python create_dataset.py --sessions ./sessions/ --out dataset.npz --in-len 25 --out-len 25 --step 1 --max-gap 0.2 --val-ratio 0.15 --seed 42