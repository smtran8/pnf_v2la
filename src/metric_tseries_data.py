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
Note that we have a 50 ms Time Syncing to match RGB with Camera Depth"""

#Edit the following to match LIMO's ros2 topic:
RGB_TOPIC = "/camera/color/image_raw" #RGB data
DEPTH_TOPIC = "/camera/depth/image_raw"    #Instead of each pixel storing R,G,B, each stores a distance number - how far away the surface at that pixel location from the camera lens, in milimeters
INFO_TOPIC = "/camera/color/camera_info" #Physical properties of the camera -fx,fy,...
#Inside CameraDepth:
#Encoding: 16UC1
#Height: 480
#Width: 640

MODEL_PATH = "/home/agilex/limo_ros2_ws/src/son_metric_projection/son_metric_projection/best.pt"
CONF = 0.3

SESSIONS_DIR = "/home/agilex/limo_ros2_ws/src/son_metric_projection/sessions/"


CROSSHAIR_COLOR = (0, 0, 255)   # red in BGR
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

#A potential issue, not every time we point a u,v is correct => It could be a noise or a small hole. Therefore we add a window size. When looks at u and v, we also look at the neighbors of u and v
#And the issue happens => For 2 or more LIMOs it will take the range of the wall behind => colliding median window
#So, let's just use the bounding box pixel instead of window median
def sample_depth(depth_image, u, v, win=5):
    """Median depth in a small window (robust to holes). Returns meters or None."""
    h, w = depth_image.shape[:2]
    u0, u1 = max(0, u - win), min(w, u + win + 1)
    v0, v1 = max(0, v - win), min(h, v + win + 1)
    patch = depth_image[v0:v1, u0:u1].astype(np.float32)
    valid = patch[patch > 0]
    if valid.size == 0:
        return None
    #Instead of median, try min to see if it does catch LIMO
    depth_mm = float(np.min(valid))
    #return depth_mm
    return depth_mm / 1000.0 
    #Need to check: If mm is already meters then we good, if not need to convert
    #Ros2 topic echo
    

class MetricProjectionNode(Node):
    def __init__(self):
        super().__init__("metric_projection")
        self.bridge = CvBridge()
        self.model = YOLO(MODEL_PATH)
        #Used a dictionary storage to send the LIMO distance back to the nearest one:
        self.last_good_metric = {}   # track_id -> (forward, side), most recent valid reading
        #self.last_frame_time = {}
        self.trajectories = {}       # track_id -> list of (timestamp, forward, side)

        # --- ROS2 parameters (set via: ros2 run ... --ros-args -p delay:=3.0 -p session:=01)
        self.declare_parameter("delay", 3.0)
        self.declare_parameter("session", "trajectory")

        # --- Recording grace period: let detection/tracking run and the live
        # window display immediately, but don't SAVE any data until
        # `delay` seconds have passed, so you have time to walk over
        # and start moving the other LIMO before real data collection begins.
        self.start_delay = self.get_parameter("delay").get_parameter_value().double_value
        session_name = self.get_parameter("session").get_parameter_value().string_value
        os.makedirs(SESSIONS_DIR, exist_ok=True)
        self.out_path = os.path.join(SESSIONS_DIR, f"session_{session_name}.json")
        self.node_start_time = time.time()
        self.recording_active = False
        self.last_countdown_shown = None   # tracks which whole-second we last announced

        # Intrinsics Parameter— filled in once from CameraInfo message 
        self.fx = self.fy = self.cx = self.cy = None
        self.create_subscription(CameraInfo, INFO_TOPIC, self.info_cb, 10)
        

        # Synchronize RGB + depth by timestamp so we always pair the right two objects
        rgb_sub = message_filters.Subscriber(self, Image, RGB_TOPIC)
        depth_sub = message_filters.Subscriber(self, Image, DEPTH_TOPIC)
        # ApproximateTime: pairs messages whose stamps are close (not identical)
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
            #Just in case:
            #1. k[0] = ~491.22
            #2. k[4] = ~491.22
            #3. k[2] = 323.98
            #4. k[5] = 213.08

    def frame_cb(self, rgb_msg: Image, depth_msg: Image):
        if self.fx is None:
            return  # wait until we have intrinsics

        # ROS Image -> OpenCV/numpy
        rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        timestamp = round(rgb_msg.header.stamp.sec + rgb_msg.header.stamp.nanosec * 1e-9, 3)

        # --- Check / flip the recording grace period, with a visible countdown ---
        if not self.recording_active:
            elapsed = time.time() - self.node_start_time
            remaining = self.start_delay - elapsed
            if remaining > 0:
                # Print once per whole second remaining, e.g. 3, 2, 1
                whole_remaining = int(remaining) + 1
                if whole_remaining != self.last_countdown_shown:
                    self.last_countdown_shown = whole_remaining
                    print(f">>> Recording starts in {whole_remaining}...")
            else:
                self.recording_active = True
                print(">>> RECORDING STARTED <<<")
                self.get_logger().info("Grace period over — recording started.")

        # Detect + track LIMOs in the RGB frame
        results = self.model.track(
            rgb, persist=True, tracker="bytetrack.yaml",
            conf=CONF, verbose=False)

        # Clean up last_good_metric entries for tracks no longer active this frame
        active_ids = {int(box.id[0]) for box in results[0].boxes if box.id is not None}
        stale_ids = set(self.last_good_metric.keys()) - active_ids
        for sid in stale_ids:
            del self.last_good_metric[sid]

        display = rgb.copy()
        for box in results[0].boxes:
            if box.id is None:
                continue
            tid = int(box.id[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            u, v = box_ground_pixel(x1, y1, x2, y2)
            u_d, v_d = scale_to_depth(u, v, rgb.shape, depth.shape)
            depth_m = sample_depth(depth, u_d, v_d)
            metric = pixel_to_metric(u, v, depth_m,
                                    self.fx, self.fy, self.cx, self.cy)

            # --- Safeguard: hold last known good value through brief depth dropouts ---
            used_fallback = False
            if metric is not None:
                self.last_good_metric[tid] = metric
                x_fwd, y_side = metric
                # Only SAVE to the trajectory buffer once the grace period has passed.
                # (Detection/tracking/display still run during the grace period;
                # we just don't persist that data into the dataset.)
                if self.recording_active:
                    self.trajectories.setdefault(tid, []).append((timestamp, x_fwd, y_side))
            elif tid in self.last_good_metric:
                metric = self.last_good_metric[tid]
                used_fallback = True

            # Draw the box + ground point regardless of depth validity
            cv2.rectangle(display, (x1, y1), (x2, y2), CROSSHAIR_COLOR, 2)
            cv2.circle(display, (u, v), 4, CROSSHAIR_COLOR, -1)
            
            #if tid in self.last_frame_time:
                #dt = timestamp - self.last_frame_time[tid]
                #print(f"ID: {tid} dt = {dt:.3f}s ~ {1/dt:.1f} HZ")
            #self.last_frame_time[tid] = timestamp

            if metric is not None:
                x_fwd, y_side = metric
                status = " (held)" if used_fallback else ""
                rec_tag = "" if self.recording_active else " [grace period - not saved]"
                self.get_logger().info(
                    f"LIMO id:{tid}  forward={x_fwd:.2f}m  side={y_side:+.2f}m{status}{rec_tag}")
                label = f"id:{tid} t = {timestamp} fwd={x_fwd:.2f}m side={y_side:+.2f}m{status}"
            else:
                self.get_logger().info(f"LIMO id:{tid}  no valid depth (no history yet)")
                label = f"id:{tid} no depth"

            cv2.putText(display, label, (x1, max(y1 - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, CROSSHAIR_COLOR, 2)

        cv2.imshow("P&F metric projection  (q to quit)", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            rclpy.shutdown()

    def save_trajectories(self):
        """Write all accumulated per-track trajectories to a JSON file.
        Called on shutdown (Ctrl+C or window 'q'). Each track_id maps to a
        list of [timestamp, forward, side] entries, in time order."""
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

#Note: Have to check if the pure Driver topic and the RGB (pixel) topic is aligned - meaning does they share the same resolution?
#ros2 topic echo /camera/color/image_raw --field width --field height --once
#ros2 topic echo /camera/depth/image_raw --field width --field height --once
#Also check:
#ros2 topic list | grep -iE "depth|aligned|registered|to_color", they might already have a depth aligned to color
#Note: when change a script file in LIMO that was already built, remember to remove all the build/ install/ and re-colcon build again:
#rm -rf build/son_metric_projection install/son_metric_projection