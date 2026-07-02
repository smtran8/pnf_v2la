import time
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import message_filters
from ultralytics import YOLO


# -----------------------------------------------------------------------
# Absolute paths
# -----------------------------------------------------------------------
MODEL_PATH = "/home/agilex/limo_ros2_ws/src/son_metric_projection/son_metric_projection/best.pt"
LSTM_PATH  = "/home/agilex/limo_ros2_ws/src/pnf_inference/pnf_inference/best_lstm.pt"

# -----------------------------------------------------------------------
# ROS2 topics
# -----------------------------------------------------------------------
RGB_TOPIC   = "/camera/color/image_raw"
DEPTH_TOPIC = "/camera/depth/image_raw"
INFO_TOPIC  = "/camera/color/camera_info"

# -----------------------------------------------------------------------
# Detection
# -----------------------------------------------------------------------
CONF = 0.3

# -----------------------------------------------------------------------
# Inference window -- must match make_dataset.py
# -----------------------------------------------------------------------
IN_LEN  = 15
OUT_LEN = 15

# -----------------------------------------------------------------------
# Camera-view display colors (BGR)
# -----------------------------------------------------------------------
COLOR_BOX  = (0,   255,  0)    # green  -- detection box
COLOR_DOT  = (0,   0,   255)   # red    -- current ground point
COLOR_PRED = (0,   255, 255)   # yellow -- predicted dots on camera view
COLOR_TEXT = (255, 255,   0)   # cyan   -- label text

# -----------------------------------------------------------------------
# Top-down panel config
# -----------------------------------------------------------------------
TD_SIZE    = 500        # canvas size in pixels (square)
TD_RANGE   = 3.5        # meters visible forward (ego LIMO to top of panel)
TD_SIDE    = 2.0        # meters visible left and right of center line
TD_SCALE_Y = TD_SIZE / TD_RANGE          # pixels per meter (forward axis)
TD_SCALE_X = TD_SIZE / (TD_SIDE * 2)    # pixels per meter (side axis)
TD_OX      = TD_SIZE // 2               # ego origin x pixel (horizontal center)
TD_OY      = TD_SIZE - 30              # ego origin y pixel (near bottom)

# Top-down colors (BGR)
TD_BG   = (20,  20,  20)    # near-black background
TD_GRID = (60,  60,  60)    # dark grey grid lines
TD_EGO  = (200, 200, 200)   # white -- ego LIMO triangle
TD_HIST = (200, 100,  50)   # blue-orange -- observed input trail
TD_CURR = (0,   255,   0)   # green -- current position
TD_PRED = (0,   255, 255)   # yellow -- predicted trajectory
TD_TXT  = (180, 180, 180)   # light grey -- labels and axis text


# -----------------------------------------------------------------------
# LSTM architecture -- must match train_lstm.py exactly
# -----------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = dropout if num_layers > 1 else 0.0,
            batch_first = True)

    def forward(self, x):
        _, (h, c) = self.lstm(x)
        return h, c


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = dropout if num_layers > 1 else 0.0,
            batch_first = True)
        self.output_layer = nn.Linear(hidden_size, input_size)

    def forward_step(self, x, h, c):
        out, (h, c) = self.lstm(x, (h, c))
        pred = self.output_layer(out)
        return pred, h, c


class EncoderDecoder(nn.Module):
    def __init__(self, input_size=2, hidden_size=96,
                 num_layers=2, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(input_size, hidden_size, num_layers, dropout)
        self.decoder = Decoder(input_size, hidden_size, num_layers, dropout)

    def predict(self, x, out_len):
        """Fully autoregressive inference -- no teacher forcing."""
        h, c = self.encoder(x)
        dec_input  = x[:, -1:, :]
        predictions = []
        for _ in range(out_len):
            pred, h, c = self.decoder.forward_step(dec_input, h, c)
            predictions.append(pred)
            dec_input = pred
        return torch.cat(predictions, dim=1)   # (1, OUT_LEN, 2)


# -----------------------------------------------------------------------
# Metric projection helpers
# -----------------------------------------------------------------------
def pixel_to_metric(u, v, depth_m, fx, fy, cx, cy):
    if depth_m is None or depth_m <= 0 or np.isnan(depth_m):
        return None
    forward = depth_m
    side    = (u - cx) * depth_m / fx
    return (forward, side)


def metric_to_pixel(forward, side, fx, fy, cx, cy):
    """Inverse projection for camera-view overlay (flat-floor approx)."""
    if forward <= 0:
        return None
    u = int(side * fx / forward + cx)
    v = int(cy)
    return u, v


def box_ground_pixel(x1, y1, x2, y2, ratio=0.4):
    u = int((x1 + x2) / 2)
    v = int(y2 - (y2 - y1) * ratio)
    return u, v


def scale_to_depth(u_color, v_color, color_shape, depth_shape):
    color_h, color_w = color_shape[:2]
    depth_h, depth_w = depth_shape[:2]
    u_depth = int(u_color * depth_w / color_w)
    v_depth = int(v_color * depth_h / color_h)
    return u_depth, v_depth


def sample_depth(depth_image, u, v, win=5):
    h, w = depth_image.shape[:2]
    u0, u1 = max(0, u - win), min(w, u + win + 1)
    v0, v1 = max(0, v - win), min(h, v + win + 1)
    patch = depth_image[v0:v1, u0:u1].astype(np.float32)
    valid = patch[patch > 0]
    if valid.size == 0:
        return None
    return float(np.min(valid)) / 1000.0


# -----------------------------------------------------------------------
# Top-down panel helpers
# -----------------------------------------------------------------------
def metric_to_topdown(forward, side):
    """
    Convert metric (forward, side) to top-down canvas pixel (px, py).

    Coordinate system:
      - Ego LIMO sits at (TD_OX, TD_OY) -- bottom center of canvas.
      - Forward increases UPWARD (py decreases as forward increases).
      - Side increases RIGHTWARD (px increases as side increases,
        negative side = left of center = smaller px).

    Returns (px, py) as integers, or None if outside canvas bounds.
    """
    px = int(TD_OX + side   * TD_SCALE_X)
    py = int(TD_OY - forward * TD_SCALE_Y)
    if 0 <= px < TD_SIZE and 0 <= py < TD_SIZE:
        return px, py
    return None


def draw_topdown_base():
    """
    Create a fresh top-down canvas with grid, axis labels, and ego marker.
    Called once per frame before drawing trajectories on top.
    """
    canvas = np.full((TD_SIZE, TD_SIZE, 3), TD_BG, dtype=np.uint8)

    # --- Grid lines every 0.5m ---
    step_fwd  = 0.5   # meters between horizontal grid lines
    step_side = 0.5   # meters between vertical grid lines

    # Horizontal lines (constant forward distance)
    fwd = step_fwd
    while fwd < TD_RANGE:
        py = int(TD_OY - fwd * TD_SCALE_Y)
        if 0 <= py < TD_SIZE:
            cv2.line(canvas, (0, py), (TD_SIZE, py), TD_GRID, 1)
            # Distance label on left edge
            cv2.putText(canvas, f"{fwd:.1f}m",
                        (4, py - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, TD_TXT, 1)
        fwd += step_fwd

        # Vertical lines (constant side offset)
    side = -TD_SIDE
    while side <= TD_SIDE + 0.01:
        px = int(TD_OX + side * TD_SCALE_X)
        if 0 <= px < TD_SIZE:
            cv2.line(canvas, (px, 0), (px, TD_SIZE), TD_GRID, 1)
            # Side label — show all including 0m, placed above ego triangle
            if abs(side) < 0.01:
                label = "0m"    # ego center line
            else:
                label = f"{side:+.1f}m"
            # Place label at a fixed y position just above the ego marker
            label_y = TD_OY - 20
            label_x = px - 14 if abs(side) > 0.1 else px - 8
            cv2.putText(canvas, label,
                        (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, TD_TXT, 1)
        side += step_side

    # Center line (side=0, straight ahead)
    cv2.line(canvas, (TD_OX, 0), (TD_OX, TD_SIZE),
             (90, 90, 90), 1)

    # --- Ego LIMO marker: small upward-pointing triangle ---
    tri_pts = np.array([
        [TD_OX,      TD_OY - 14],   # tip (front of ego)
        [TD_OX - 8,  TD_OY + 4],    # bottom left
        [TD_OX + 8,  TD_OY + 4],    # bottom right
    ], np.int32)
    cv2.fillPoly(canvas, [tri_pts], TD_EGO)

    # --- Title ---
    cv2.putText(canvas, "Top-Down View  (ego=triangle)",
                (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, TD_TXT, 1)

    # --- Legend ---
    # --- Legend (top-right, between 2.5m and 3.0m grid lines) ---
    legend_x = TD_SIZE - 160   # right side of canvas
    legend_y = 35              # just below the title, above 3.0m line
    cv2.circle(canvas, (legend_x, legend_y),      5, TD_HIST, -1)
    cv2.putText(canvas, "History (input)",
                (legend_x + 10, legend_y + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, TD_TXT, 1)
    cv2.circle(canvas, (legend_x, legend_y + 18), 5, TD_CURR, -1)
    cv2.putText(canvas, "Current position",
                (legend_x + 10, legend_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, TD_TXT, 1)
    cv2.circle(canvas, (legend_x, legend_y + 36), 5, TD_PRED, -1)
    cv2.putText(canvas, "Predicted (output)",
                (legend_x + 10, legend_y + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, TD_TXT, 1)

    return canvas


# -----------------------------------------------------------------------
# Inference node
# -----------------------------------------------------------------------
class PnFInferenceNode(Node):
    def __init__(self):
        super().__init__("pnf_inference")
        self.bridge = CvBridge()

        # --- ROS2 parameters ---
        self.declare_parameter("delay", 2.0)
        self.start_delay = (self.get_parameter("delay")
                               .get_parameter_value().double_value)

        # --- Device selection FIRST ---
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.get_logger().info(f"Device: {self.device}")

        # --- YOLO detector ---
        self.detector = YOLO(MODEL_PATH)
        self.detector.to(self.device)
        self.get_logger().info(f"Detector device: {self.detector.device}")

        # --- LSTM ---
        self.lstm = EncoderDecoder(input_size=2, hidden_size=96,
                                   num_layers=2, dropout=0.1)
        self.lstm.load_state_dict(
            torch.load(LSTM_PATH, map_location=self.device))
        self.lstm.to(self.device)
        self.lstm.eval()
        self.get_logger().info(f"LSTM loaded from: {LSTM_PATH}")
        self.get_logger().info(f"LSTM device: {self.device}")

        # --- Grace period ---
        self.node_start_time      = time.time()
        self.inference_active     = False
        self.last_countdown_shown = None

        # --- Per-track state ---
        self.buffers          = {}   # track_id -> deque(maxlen=IN_LEN)
        self.last_good_metric = {}   # track_id -> (forward, side) fallback
        self.last_prediction  = {}   # track_id -> (OUT_LEN, 2) numpy array

        # --- Camera intrinsics ---
        self.fx = self.fy = self.cx = self.cy = None
        self.create_subscription(CameraInfo, INFO_TOPIC, self.info_cb, 10)

        # --- Synchronized RGB + depth ---
        rgb_sub   = message_filters.Subscriber(self, Image, RGB_TOPIC)
        depth_sub = message_filters.Subscriber(self, Image, DEPTH_TOPIC)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub], queue_size=10, slop=0.05)
        self.sync.registerCallback(self.frame_cb)

        self.get_logger().info("P&F inference node ready.")
        self.get_logger().info(
            f"Inference begins in {self.start_delay:.1f}s -- "
            f"drive the other LIMO during the countdown.")

    # ------------------------------------------------------------------
    def info_cb(self, msg: CameraInfo):
        if self.fx is None:
            self.fx = round(msg.k[0], 2)
            self.fy = round(msg.k[4], 2)
            self.cx = round(msg.k[2], 2)
            self.cy = round(msg.k[5], 2)
            self.get_logger().info(
                f"Intrinsics: fx={self.fx} fy={self.fy} "
                f"cx={self.cx} cy={self.cy}")

    # ------------------------------------------------------------------
    def frame_cb(self, rgb_msg: Image, depth_msg: Image):
        if self.fx is None:
            return

        rgb   = self.bridge.imgmsg_to_cv2(rgb_msg,   desired_encoding="bgr8")
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")

        # --- Grace period countdown ---
        if not self.inference_active:
            elapsed   = time.time() - self.node_start_time
            remaining = self.start_delay - elapsed
            if remaining > 0:
                whole = int(remaining) + 1
                if whole != self.last_countdown_shown:
                    self.last_countdown_shown = whole
                    self.get_logger().info(
                        f">>> Inference starts in {whole}...")
            else:
                self.inference_active = True
                self.get_logger().info(">>> INFERENCE STARTED <<<")

        # --- Detection + tracking ---
        results = self.detector.track(
            rgb, persist=True, tracker="bytetrack.yaml",
            conf=CONF, verbose=False)

        # Clean up stale tracks
        active_ids = {int(b.id[0]) for b in results[0].boxes
                      if b.id is not None}
        for sid in set(self.buffers.keys()) - active_ids:
            del self.buffers[sid]
        for sid in set(self.last_good_metric.keys()) - active_ids:
            del self.last_good_metric[sid]
        for sid in set(self.last_prediction.keys()) - active_ids:
            del self.last_prediction[sid]

        display  = rgb.copy()
        topdown  = draw_topdown_base()   # fresh canvas every frame

        for box in results[0].boxes:
            if box.id is None:
                continue
            tid = int(box.id[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            u, v     = box_ground_pixel(x1, y1, x2, y2)
            u_d, v_d = scale_to_depth(u, v, rgb.shape, depth.shape)
            depth_m  = sample_depth(depth, u_d, v_d)
            metric   = pixel_to_metric(u, v, depth_m,
                                       self.fx, self.fy, self.cx, self.cy)

            # Held-value safeguard
            used_fallback = False
            if metric is not None:
                self.last_good_metric[tid] = metric
            elif tid in self.last_good_metric:
                metric        = self.last_good_metric[tid]
                used_fallback = True

            if metric is None:
                continue

            x_fwd, y_side = metric

            # --- Update sliding window buffer ---
            if tid not in self.buffers:
                self.buffers[tid] = deque(maxlen=IN_LEN)
            if self.inference_active:
                self.buffers[tid].append((x_fwd, y_side))

            # --- Run LSTM once buffer is full ---
            prediction = None
            if self.inference_active and len(self.buffers[tid]) == IN_LEN:
                prediction = self._run_lstm(self.buffers[tid])
                self.last_prediction[tid] = prediction
                status = " (held)" if used_fallback else ""
                self.get_logger().info(
                    f"LIMO id:{tid}  "
                    f"fwd={x_fwd:.2f}m  side={y_side:+.2f}m{status}  "
                    f"-> pred[0]: fwd={prediction[0,0]:.2f}m "
                    f"side={prediction[0,1]:+.2f}m")
            elif tid in self.last_prediction:
                prediction = self.last_prediction[tid]

            # --- Camera view: box + ground point + predicted dots ---
            cv2.rectangle(display, (x1, y1), (x2, y2), COLOR_BOX, 2)
            cv2.circle(display, (u, v), 5, COLOR_DOT, -1)

            n_buf = len(self.buffers.get(tid, []))
            label = (f"id:{tid}  fwd={x_fwd:.2f}m  "
                     f"side={y_side:+.2f}m  buf={n_buf}/{IN_LEN}")
            cv2.putText(display, label, (x1, max(y1 - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1)

            if prediction is not None:
                self._draw_camera_prediction(display, prediction)

            # --- Top-down view: history trail + current + predicted ---
            self._draw_topdown_track(topdown, tid, x_fwd, y_side, prediction)

        # --- Grace period overlay ---
        if not self.inference_active:
            elapsed   = time.time() - self.node_start_time
            remaining = max(0.0, self.start_delay - elapsed)
            cv2.putText(display,
                        f"Starting in {remaining:.1f}s ...",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, COLOR_PRED, 2)
            cv2.putText(topdown,
                        f"Starting in {remaining:.1f}s ...",
                        (TD_SIZE//2 - 80, TD_SIZE//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, TD_PRED, 2)

        cv2.imshow("P&F Camera View  (q to quit)", display)
        cv2.imshow("P&F Top-Down View", topdown)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            rclpy.shutdown()

    # ------------------------------------------------------------------
    def _run_lstm(self, buffer: deque) -> np.ndarray:
        """Run one LSTM forward pass. Returns (OUT_LEN, 2) numpy array."""
        x = (torch.tensor(list(buffer), dtype=torch.float32)
                  .unsqueeze(0)
                  .to(self.device))
        with torch.no_grad():
            pred = self.lstm.predict(x, OUT_LEN)
        return pred.squeeze(0).cpu().numpy()

    # ------------------------------------------------------------------
    def _draw_camera_prediction(self, display: np.ndarray,
                                 prediction: np.ndarray) -> None:
        """Draw predicted trajectory as fading dots on the camera view."""
        h, w = display.shape[:2]
        for k, (fwd, side) in enumerate(prediction):
            px = metric_to_pixel(fwd, side,
                                 self.fx, self.fy, self.cx, self.cy)
            if px is None:
                continue
            pu, pv = px
            if not (0 <= pu < w and 0 <= pv < h):
                continue
            radius = max(3, 8 - k // 2)
            alpha  = 1.0 - (k / OUT_LEN) * 0.6
            color  = tuple(int(c * alpha) for c in COLOR_PRED)
            cv2.circle(display, (pu, pv), radius, color, -1)
            if k > 0:
                prev_fwd, prev_side = prediction[k - 1]
                prev_px = metric_to_pixel(prev_fwd, prev_side,
                                          self.fx, self.fy,
                                          self.cx, self.cy)
                if prev_px is not None:
                    pu_prev, pv_prev = prev_px
                    if 0 <= pu_prev < w and 0 <= pv_prev < h:
                        cv2.line(display, (pu_prev, pv_prev),
                                 (pu, pv), color, 1)

    # ------------------------------------------------------------------
    def _draw_topdown_track(self, canvas: np.ndarray,
                             tid: int,
                             x_fwd: float, y_side: float,
                             prediction) -> None:
        """
        Draw one track's history, current position, and prediction
        onto the top-down canvas.

        History trail (blue): the IN_LEN buffered positions, fading
          from oldest (dim) to newest (bright).
        Current position (green): large dot at the most recent reading.
        Predicted trajectory (yellow): OUT_LEN predicted future positions,
          fading from bright (near future) to dim (far future).
          Every 5th step is labeled (t+5, t+10, t+15).
        """
        # --- History trail ---
        history = list(self.buffers.get(tid, []))
        prev_hp = None
        for i, (hfwd, hside) in enumerate(history):
            hp = metric_to_topdown(hfwd, hside)
            if hp is None:
                prev_hp = None
                continue
            # Fade: oldest = dim (0.2), newest = bright (0.9)
            alpha  = 0.2 + 0.7 * (i / max(len(history) - 1, 1))
            color  = tuple(int(c * alpha) for c in TD_HIST)
            radius = 3 if i < len(history) - 1 else 5
            cv2.circle(canvas, hp, radius, color, -1)
            if prev_hp is not None:
                cv2.line(canvas, prev_hp, hp, color, 1)
            prev_hp = hp

        # --- Current position (last in history = "now") ---
        curr_p = metric_to_topdown(x_fwd, y_side)
        if curr_p is not None:
            cv2.circle(canvas, curr_p, 7, TD_CURR, -1)
            cv2.putText(canvas, f"id:{tid}",
                        (curr_p[0] + 8, curr_p[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, TD_CURR, 1)

        # --- Predicted trajectory ---
        if prediction is None:
            return

        prev_pp = curr_p   # connect first predicted dot to current position
        for k, (pfwd, pside) in enumerate(prediction):
            pp = metric_to_topdown(pfwd, pside)
            if pp is None:
                prev_pp = None
                continue

            # Fade: near future bright (1.0), far future dim (0.3)
            alpha  = 1.0 - (k / OUT_LEN) * 0.7
            color  = tuple(int(c * alpha) for c in TD_PRED)
            radius = max(2, 6 - k // 3)

            cv2.circle(canvas, pp, radius, color, -1)

            # Connect to previous dot with a line
            if prev_pp is not None:
                cv2.line(canvas, prev_pp, pp, color, 1)

            # Label every 5th step: t+5, t+10, t+15
            if (k + 1) % 5 == 0:
                cv2.putText(canvas, f"t+{k+1}",
                            (pp[0] + 5, pp[1] - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, TD_TXT, 1)

            prev_pp = pp


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    rclpy.init()
    node = PnFInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

# Setup instructions:
# 1. Copy this file to:
#    ~/limo_ros2_ws/src/pnf_inference/pnf_inference/inference.py
# 2. Copy best_lstm.pt to:
#    ~/limo_ros2_ws/src/pnf_inference/pnf_inference/best_lstm.pt
# 3. setup.py entry_points:
#    'pnf_inference = pnf_inference.inference:main'
# 4. Rebuild:
#    rm -rf build/pnf_inference install/pnf_inference
#    colcon build --packages-select pnf_inference
#    source install/setup.bash
# 5. Run:
#    ros2 run pnf_inference pnf_inference --ros-args -p delay:=2.0