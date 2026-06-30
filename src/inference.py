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
# Absolute paths — confirmed for this LIMO
# -----------------------------------------------------------------------
MODEL_PATH = "/home/agilex/limo_ros2_ws/src/pnf_inference/pnf_inference/best.pt"
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
# Inference window — must match make_dataset.py
# -----------------------------------------------------------------------
IN_LEN  = 15
OUT_LEN = 15

# -----------------------------------------------------------------------
# Display colors (BGR)
# -----------------------------------------------------------------------
COLOR_BOX  = (0,   255,  0)    # green  — detection box
COLOR_DOT  = (0,   0,   255)   # red    — current ground point
COLOR_PRED = (0,   255, 255)   # yellow — predicted future positions
COLOR_TEXT = (255, 255,   0)   # cyan   — label text


# -----------------------------------------------------------------------
# LSTM architecture — must match train_lstm.py exactly
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
        """Fully autoregressive inference — no teacher forcing."""
        h, c = self.encoder(x)
        dec_input  = x[:, -1:, :]   # seed with last observed position
        predictions = []
        for _ in range(out_len):
            pred, h, c = self.decoder.forward_step(dec_input, h, c)
            predictions.append(pred)
            dec_input = pred
        return torch.cat(predictions, dim=1)   # (1, OUT_LEN, 2)


# -----------------------------------------------------------------------
# Metric projection helpers — identical to metric_projection_node.py
# -----------------------------------------------------------------------
def pixel_to_metric(u, v, depth_m, fx, fy, cx, cy):
    if depth_m is None or depth_m <= 0 or np.isnan(depth_m):
        return None
    forward = depth_m
    side    = (u - cx) * depth_m / fx
    return (forward, side)


def metric_to_pixel(forward, side, fx, fy, cx, cy):
    """
    Inverse of pixel_to_metric.
    Projects a predicted (forward, side) ground-plane position back to
    image pixel coordinates using the flat-floor approximation (v = cy).
    Returns (u, v) as integers, or None if forward <= 0.
    """
    if forward <= 0:
        return None
    u = int(side * fx / forward + cx)
    v = int(cy)   # flat-floor: all ground points sit at the horizon line
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

        # --- Device selection FIRST — before anything that needs it ---
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

        # --- Grace period state ---
        self.node_start_time      = time.time()
        self.inference_active     = False
        self.last_countdown_shown = None

        # --- Per-track state ---
        # deque(maxlen=IN_LEN) auto-drops oldest reading when full —
        # gives the 0-14, 1-15, 2-16 sliding window automatically.
        self.buffers          = {}   # track_id -> deque of (forward, side)
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
            f"Inference begins in {self.start_delay:.1f}s — "
            f"drive the other LIMO during the countdown.")

    # ------------------------------------------------------------------
    def info_cb(self, msg: CameraInfo):
        if self.fx is None:
            self.fx = round(msg.k[0], 2)
            self.fy = round(msg.k[4], 2)
            self.cx = round(msg.k[2], 2)
            self.cy = round(msg.k[5], 2)
            #self.get_logger().info(
                #f"Intrinsics: fx={self.fx} fy={self.fy} "
                #f"cx={self.cx} cy={self.cy}")

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

        display = rgb.copy()

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

            # --- Draw detection box + ground point ---
            cv2.rectangle(display, (x1, y1), (x2, y2), COLOR_BOX, 2)
            cv2.circle(display, (u, v), 5, COLOR_DOT, -1)

            # --- Label ---
            n_buf = len(self.buffers.get(tid, []))
            label = (f"id:{tid}  fwd={x_fwd:.2f}m  "
                     f"side={y_side:+.2f}m  buf={n_buf}/{IN_LEN}")
            cv2.putText(display, label, (x1, max(y1 - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1)

            # --- Draw predicted trajectory ---
            if prediction is not None:
                self._draw_prediction(display, prediction)

        # --- Grace period overlay on window ---
        if not self.inference_active:
            elapsed   = time.time() - self.node_start_time
            remaining = max(0.0, self.start_delay - elapsed)
            cv2.putText(display,
                        f"Starting in {remaining:.1f}s ...",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, COLOR_PRED, 2)

        cv2.imshow("P&F Live Inference  (q to quit)", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            rclpy.shutdown()

    # ------------------------------------------------------------------
    def _run_lstm(self, buffer: deque) -> np.ndarray:
        """
        One LSTM forward pass on the current buffer.
        buffer: deque of IN_LEN (forward, side) tuples
        Returns: numpy array shape (OUT_LEN, 2)
        """
        x = (torch.tensor(list(buffer), dtype=torch.float32)
                  .unsqueeze(0)
                  .to(self.device))
        with torch.no_grad():
            pred = self.lstm.predict(x, OUT_LEN)   # (1, OUT_LEN, 2)
        return pred.squeeze(0).cpu().numpy()        # (OUT_LEN, 2)

    # ------------------------------------------------------------------
    def _draw_prediction(self, display: np.ndarray,
                          prediction: np.ndarray) -> None:
        """
        Project each predicted (forward, side) back to pixel space
        and draw as fading dots connected by lines.

        Dots fade in size (8 → 3px) and brightness (100% → 40%)
        from near-future to far-future to show time progression.
        """
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

            # Connect to previous dot
            if k > 0:
                prev_fwd, prev_side = prediction[k - 1]
                prev_px = metric_to_pixel(prev_fwd, prev_side,
                                          self.fx, self.fy,
                                          self.cx, self.cy)
                if prev_px is not None:
                    pu_prev, pv_prev = prev_px
                    if 0 <= pu_prev < w and 0 <= pv_prev < h:
                        cv2.line(display,
                                 (pu_prev, pv_prev), (pu, pv),
                                 color, 1)


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

