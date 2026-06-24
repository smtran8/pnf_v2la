import json
import os
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import message_filters
from ultralytics import YOLO
import cv2
"""Metric Projection:
Subscribes to synchronized RGB + depth from the LIMO's Orbbec camera, runs
YOLO+ByteTrack on the RGB, looks up depth at each tracked LIMO's ground point,
and projects to metric (x, y). Publishes/prints the per-track metric positions.
Note that we have a 50 ms Time Syncing to match RGB with Camera Depth

Display is disabled for data collection mode.
To re-enable: uncomment the display block marked with # [DISPLAY] below.
"""

#Edit the following to match LIMO's ros2 topic:
RGB_TOPIC = "/camera/color/image_raw"
DEPTH_TOPIC = "/camera/depth/image_raw"
INFO_TOPIC = "/camera/color/camera_info"
#Inside CameraDepth:
#Encoding: 16UC1
#Height: 480
#Width: 640

MODEL_PATH = "/home/agilex/limo_ros2_ws/src/son_metric_projection/son_metric_projection/best.pt"
CONF = 0.3

SESSIONS_DIR = "/home/agilex/limo_ros2_ws/src/son_metric_projection/sessions/"

CROSSHAIR_COLOR = (0, 0, 255)   # red in BGR — kept for when display is re-enabled
CROSSHAIR_THICK = 2


def pixel_to_metric(u, v, depth_m, fx, fy, cx, cy):
    """calculate the camera forward and sideway distance to the obstacle"""
    if depth_m is None or depth_m <= 0 or np.isnan(depth_m):
        return None
    forward = depth_m                    # camera Z axis: straight ahead
    side    = (u - cx) * depth_m / fx    # camera X axis: left/right
    return (forward, side)

def box_ground_pixel(x1, y1, x2, y2, ratio=0.4):
    """Sample point pulled up 40% of the box height from the bottom,
    away from the noisy/edge-of-frame region near the true ground contact."""
    u = int((x1 + x2) / 2)
    v = int(y2 - (y2 - y1) * ratio)
    return u, v

def scale_to_depth(u_color, v_color, color_shape, depth_shape):
    """Convert color-space pixel to depth-space pixel, for SAMPLING ONLY."""
    color_h, color_w = color_shape[:2]
    depth_h, depth_w = depth_shape[:2]
    u_depth = int(u_color * depth_w / color_w)
    v_depth = int(v_color * depth_h / color_h)
    return u_depth, v_depth

def sample_depth(depth_image, u, v, win=5):
    """Min depth in a small window (robust to holes). Returns meters or None."""
    h, w = depth_image.shape[:2]
    u0, u1 = max(0, u - win), min(w, u + win + 1)
    v0, v1 = max(0, v - win), min(h, v + win + 1)
    patch = depth_image[v0:v1, u0:u1].astype(np.float32)
    valid = patch[patch > 0]
    if valid.size == 0:
        return None
    depth_mm = float(np.min(valid))
    return depth_mm / 1000.0


class MetricProjectionNode(Node):
    def __init__(self):
        super().__init__("metric_projection")
        self.bridge = CvBridge()
        self.model = YOLO(MODEL_PATH)
        self.last_good_metric = {}   # track_id -> (forward, side)
        self.trajectories = {}       # track_id -> list of (timestamp, forward, side)

        # --- ROS2 parameters (set via: ros2 run ... --ros-args -p delay:=3.0 -p session:=01)
        self.declare_parameter("delay", 3.0)
        self.declare_parameter("session", 888)

        self.start_delay = self.get_parameter("delay").get_parameter_value().double_value
        session_num = self.get_parameter("session").get_parameter_value().integer_value
        os.makedirs(SESSIONS_DIR, exist_ok=True)
        self.out_path = os.path.join(SESSIONS_DIR, f"session_{session_num:02d}.json")
        self.node_start_time = time.time()
        self.recording_active = False
        self.last_countdown_shown = None

        self.fx = self.fy = self.cx = self.cy = None
        self.create_subscription(CameraInfo, INFO_TOPIC, self.info_cb, 10)

        rgb_sub   = message_filters.Subscriber(self, Image, RGB_TOPIC)
        depth_sub = message_filters.Subscriber(self, Image, DEPTH_TOPIC)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub], queue_size=10, slop=0.05)
        self.sync.registerCallback(self.frame_cb)

        self.get_logger().info("Metric projection node started. Waiting for frames...")
        self.get_logger().info(
            f"Recording will begin {self.start_delay:.1f}s after first frame; "
            f"saving to '{self.out_path}' on shutdown.")

    def info_cb(self, msg: CameraInfo):
        """Grab intrinsics once from the K matrix"""
        if self.fx is None:
            self.fx = round(msg.k[0], 2)
            self.fy = round(msg.k[4], 2)
            self.cx = round(msg.k[2], 2)
            self.cy = round(msg.k[5], 2)
            self.get_logger().info(
                f"Got intrinsics: fx={self.fx:.1f} fy={self.fy:.1f} "
                f"cx={self.cx:.1f} cy={self.cy:.1f}")
            #1. k[0] = ~491.22
            #2. k[4] = ~491.22
            #3. k[2] = 323.98
            #4. k[5] = 213.08

    def frame_cb(self, rgb_msg: Image, depth_msg: Image):
        if self.fx is None:
            return

        rgb   = self.bridge.imgmsg_to_cv2(rgb_msg,   desired_encoding="bgr8")
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        timestamp = round(
            rgb_msg.header.stamp.sec + rgb_msg.header.stamp.nanosec * 1e-9, 3)

        # --- Grace period countdown ---
        if not self.recording_active:
            elapsed   = time.time() - self.node_start_time
            remaining = self.start_delay - elapsed
            if remaining > 0:
                whole_remaining = int(remaining) + 1
                if whole_remaining != self.last_countdown_shown:
                    self.last_countdown_shown = whole_remaining
                    print(f">>> Recording starts in {whole_remaining}...")
            else:
                self.recording_active = True
                print(">>> RECORDING STARTED <<<")
                self.get_logger().info("Grace period over — recording started.")

        results = self.model.track(
            rgb, persist=True, tracker="bytetrack.yaml",
            conf=CONF, verbose=False)

        # Clean up stale tracks
        active_ids = {int(box.id[0]) for box in results[0].boxes if box.id is not None}
        stale_ids  = set(self.last_good_metric.keys()) - active_ids
        for sid in stale_ids:
            del self.last_good_metric[sid]

        # [DISPLAY] Uncomment to re-enable the live window:
        # display = rgb.copy()

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

            # Safeguard: hold last known good value through brief depth dropouts
            used_fallback = False
            if metric is not None:
                self.last_good_metric[tid] = metric
                x_fwd, y_side = metric
                if self.recording_active:
                    self.trajectories.setdefault(tid, []).append(
                        (timestamp, x_fwd, y_side))
            elif tid in self.last_good_metric:
                metric        = self.last_good_metric[tid]
                used_fallback = True

            # [DISPLAY] Uncomment to re-enable drawing:
            # cv2.rectangle(display, (x1, y1), (x2, y2), CROSSHAIR_COLOR, 2)
            # cv2.circle(display, (u, v), 4, CROSSHAIR_COLOR, -1)
            # if metric is not None:
            #     x_fwd, y_side = metric
            #     status = " (held)" if used_fallback else ""
            #     label = (f"id:{tid} t={timestamp} "
            #              f"fwd={x_fwd:.2f}m side={y_side:+.2f}m{status}")
            # else:
            #     label = f"id:{tid} no depth"
            # cv2.putText(display, label, (x1, max(y1 - 8, 12)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.55, CROSSHAIR_COLOR, 2)

            if metric is not None:
                x_fwd, y_side = metric
                status  = " (held)" if used_fallback else ""
                rec_tag = "" if self.recording_active else " [grace period - not saved]"
                self.get_logger().info(
                    f"LIMO id:{tid}  forward={x_fwd:.2f}m  "
                    f"side={y_side:+.2f}m{status}{rec_tag}")
            else:
                self.get_logger().info(
                    f"LIMO id:{tid}  no valid depth (no history yet)")

        # [DISPLAY] Uncomment to re-enable the window:
        # cv2.imshow("P&F metric projection  (q to quit)", display)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     cv2.destroyAllWindows()
        #     rclpy.shutdown()

    def save_trajectories(self):
        """Write all accumulated per-track trajectories to a JSON file."""
        data = {str(tid): traj for tid, traj in self.trajectories.items()}
        with open(self.out_path, "w") as f:
            json.dump(data, f, indent=2)
        n_points = sum(len(traj) for traj in self.trajectories.values())
        self.get_logger().info(
            f"Saved {len(self.trajectories)} track(s), {n_points} total points, "
            f"to '{self.out_path}'")


def main():
    rclpy.init()
    node = MetricProjectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_trajectories()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

#Note: when change a script file in LIMO that was already built, remember to:
#rm -rf build/son_metric_projection install/son_metric_projection
#colcon build --packages-select son_metric_projection
#source install/setup.bash