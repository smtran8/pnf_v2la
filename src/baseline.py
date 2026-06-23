import argparse
import json
import glob
import os
import numpy as np


OUT_LEN   = 25
STEP      = 1
MAX_GAP_S = 0.2


def load_session(path: str) -> dict:
    """Load one session JSON. Returns {track_id (int): [(t, fwd, side), ...]}"""
    with open(path) as f:
        raw = json.load(f)
    return {int(tid): [tuple(pt) for pt in readings]
            for tid, readings in raw.items()}


def make_windows_with_timestamps(trajectory, in_len, out_len, step, max_gap_s):
    """
    Slide an (in_len + out_len) window along one track's trajectory.
    Keeps timestamps (needed for dynamic dt in the velocity calculation).
    Discards windows containing timestamp gaps > max_gap_s.
    Returns list of (input_chunk, target_chunk), each a list of (t, fwd, side).
    """
    total = in_len + out_len
    n = len(trajectory)
    if n < total:
        return []

    windows = []
    for start in range(0, n - total + 1, step):
        chunk = trajectory[start : start + total]
        timestamps = [pt[0] for pt in chunk]
        gaps = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        if max(gaps) > max_gap_s:
            continue
        windows.append((chunk[:in_len], chunk[in_len:]))
    return windows


def constant_velocity_predict(input_chunk, out_len):
    """
    Core prediction: extrapolate from the last 2 input positions.

    velocity = (position[t] - position[t-1]) / dt
    predicted[k] = position[t] + velocity * (k+1) * dt
        for k = 0, 1, ..., out_len-1

    dt is computed dynamically from the last 2 timestamps in input_chunk.

    Returns:
        predicted : np.ndarray shape (out_len, 2) — [forward, side]
    """
    t_prev, fwd_prev, side_prev = input_chunk[-2]
    t_curr, fwd_curr, side_curr = input_chunk[-1]

    dt = t_curr - t_prev
    if dt <= 0:
        dt = 1e-3   # guard against identical timestamps

    pos = np.array([fwd_curr,  side_curr],  dtype=np.float64)
    vel = np.array([
        (fwd_curr  - fwd_prev)  / dt,
        (side_curr - side_prev) / dt
    ], dtype=np.float64)

    predicted = np.stack([pos + vel * (k + 1) * dt for k in range(out_len)])
    return predicted.astype(np.float32)


def displacement_error(predicted, actual):
    """
    L2 (Euclidean) distance at each timestep — the standard metric.
    Returns (out_len,) array of per-step distances in meters.
    """
    return np.linalg.norm(predicted - actual, axis=1)


def component_error(predicted, actual):
    """
    Absolute error in each direction separately.
    Returns (fwd_errors, side_errors), each shape (out_len,).
    Useful for diagnosing which direction the model drifts more.
    """
    #Grab the absolute value here => calculate aggregated mean later
    diff = np.abs(predicted - actual)
    return diff[:, 0], diff[:, 1]


def evaluate(sessions_dir, in_len, out_len, step, max_gap_s):
    json_files = sorted(glob.glob(os.path.join(sessions_dir, "*.json")))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in: {sessions_dir}")

    # Accumulators — one value per window, pooled across all sessions/tracks
    all_ade       = []
    all_fde       = []
    all_fwd_ade   = []
    all_side_ade  = []
    all_fwd_fde   = []
    all_side_fde  = []

    print(f"Constant-Velocity Baseline")
    print(f"Window: {in_len}-in / {out_len}-out  |  "
          f"step={step}  |  max_gap={max_gap_s}s")
    print(f"{'─'*65}")

    # Collect per-session summary lines first, then print selectively
    session_summaries = []

    for path in json_files:
        session_name = os.path.basename(path)
        session_data = load_session(path)

        s_ade      = []
        s_fde      = []
        s_windows  = 0

        for tid, trajectory in session_data.items():
            windows = make_windows_with_timestamps(
                trajectory, in_len, out_len, step, max_gap_s)

            for inp_chunk, tgt_chunk in windows:
                predicted = constant_velocity_predict(inp_chunk, out_len)
                actual    = np.array([[pt[1], pt[2]] for pt in tgt_chunk],
                                     dtype=np.float32)

                # L2 displacement
                errors = displacement_error(predicted, actual)
                ade    = float(np.mean(errors))
                fde    = float(errors[-1])

                # Component-wise
                fwd_err, side_err = component_error(predicted, actual)

                s_ade.append(ade)
                s_fde.append(fde)
                s_windows += 1

                all_ade.append(ade)
                all_fde.append(fde)
                all_fwd_ade.append(float(np.mean(fwd_err)))
                all_side_ade.append(float(np.mean(side_err)))
                all_fwd_fde.append(float(fwd_err[-1]))
                all_side_fde.append(float(side_err[-1]))

        if s_windows > 0:
            session_summaries.append(
                f"  {session_name:<32}  windows={s_windows:4d}  "
                f"ADE={np.mean(s_ade):.4f}m  FDE={np.mean(s_fde):.4f}m")
        else:
            session_summaries.append(
                f"  {session_name:<32}  no valid windows")

    # Print only first 2 and last 2 session lines to keep terminal clean
    n_sessions = len(session_summaries)
    if n_sessions <= 4:
        for line in session_summaries:
            print(line)
    else:
        for line in session_summaries[:2]:
            print(line)
        print(f"  ... ({n_sessions - 4} sessions omitted) ...")
        for line in session_summaries[-2:]:
            print(line)

    # ── Overall summary ──────────────────────────────────────────────────
    print(f"{'─'*65}")
    if not all_ade:
        print("No valid windows found across all sessions.")
        return

    n = len(all_ade)
    W = 32  # column width for labels

    print(f"\n  OVERALL  ({n} windows across {len(json_files)} session(s))\n")
    print(f"  {'Metric':<{W}} {'ADE':>10} {'FDE':>10}")
    print(f"  {'─'*52}")
    print(f"  {'L2 Displacement  (standard)':<{W}} "
          f"{np.mean(all_ade):>10.4f}m {np.mean(all_fde):>10.4f}m")
    print(f"  {'Forward error only':<{W}} "
          f"{np.mean(all_fwd_ade):>10.4f}m {np.mean(all_fwd_fde):>10.4f}m")
    print(f"  {'Side error only':<{W}} "
          f"{np.mean(all_side_ade):>10.4f}m {np.mean(all_side_fde):>10.4f}m")


def main():
    p = argparse.ArgumentParser(
        description="P&F constant-velocity baseline — evaluate on session JSONs")
    p.add_argument("--sessions", required=True,
                   help="folder containing session_*.json files from Job A")
    p.add_argument("--in-len",  type=int,   default=IN_LEN)
    p.add_argument("--out-len", type=int,   default=OUT_LEN)
    p.add_argument("--step",    type=int,   default=STEP)
    p.add_argument("--max-gap", type=float, default=MAX_GAP_S)
    args = p.parse_args()

    evaluate(args.sessions, args.in_len, args.out_len,
             args.step, args.max_gap)


if __name__ == "__main__":
    main()