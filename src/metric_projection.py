import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import message_filters
from ultralytics import YOLO

"""Metric Projection:
Subscribes to synchronized RGB + depth from the LIMO's Orbbec camera, runs
YOLO+ByteTrack on the RGB, looks up depth at each tracked LIMO's ground point,
and projects to metric (x, y). Publishes/prints the per-track metric positions.
Note that we have a 50 ms Time Syncing to match RGB with Camera Depth"""

#Edit the following to match LIMO's ros2 topic:
RGB_TOPIC = "/camera/color/image_raw"
DEPTH_TOPIC = "/camera/depth/image_raw"     
INFO_TOPIC = "/camera/color/camera_info"

MODEL_PATH = "best.pt"
CONF = 0.3


def pixel_to_metric(u, v, depth_m, fx, fy, cx, cy):
    """calculate the camera forward and sideway distance to the obstacle"""
    if depth_m is None or depth_m <= 0 or np.isnan(depth_m):
        return None
    forward = depth_m                    # camera Z axis: straight ahead
    side    = (u - cx) * depth_m / fx    # camera X axis: left/right
    return (forward, side)

def box_ground_pixel(x1, y1, x2, y2):
    """Compute u and v => Bottom-center pixel, which is also ground contact point"""
    return int((x1 + x2) / 2), int(y2)


#A potential issue, not every time we point a u,v is correct => It could be a noise or a small hole. Therefore we add a window size. When looks at u and v, we also look at the neighbors of u and v

def sample_depth(depth_image, u, v, win=5):
    """Median depth in a small window (robust to holes). Returns meters or None."""
    h, w = depth_image.shape[:2]
    u0, u1 = max(0, u - win), min(w, u + win + 1)
    v0, v1 = max(0, v - win), min(h, v + win + 1)
    patch = depth_image[v0:v1, u0:u1].astype(np.float32)
    valid = patch[patch > 0]
    if valid.size == 0:
        return None
    depth_mm = float(np.median(valid))
    return depth_mm
    #return depth_mm / 1000.0 
    #Need to check: If mm is already meters then we good, if not need to convert
    #Ros2 topic echo
    
    
class MetricProjectionNode(Node):
    def __init__(self):
        super().__init__("metric_projection")
        self.bridge = CvBridge()
        self.model = YOLO(MODEL_PATH)
 
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
 
    def info_cb(self, msg: CameraInfo):
        """Grab intrinsics once from the K matrix"""
        if self.fx is None:
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]
            self.get_logger().info(
                f"Got intrinsics: fx={self.fx:.1f} fy={self.fy:.1f} "
                f"cx={self.cx:.1f} cy={self.cy:.1f}")
 
    def frame_cb(self, rgb_msg: Image, depth_msg: Image):
        if self.fx is None:
            return  # wait until we have intrinsics
 
        # ROS Image -> OpenCV/numpy
        rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
 
        results = self.model.track(
            rgb, persist=True, tracker="bytetrack.yaml",
            conf=CONF, verbose=False)
 
        for box in results[0].boxes:
            if box.id is None:
                continue
            tid = int(box.id[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            u, v = box_ground_pixel(x1, y1, x2, y2)
 
            depth_m = sample_depth(depth, u, v)
            metric = pixel_to_metric(u, v, depth_m,
                                     self.fx, self.fy, self.cx, self.cy)
 
            if metric is not None:
                x_fwd, y_side = metric
                self.get_logger().info(
                    f"LIMO id:{tid}  x={x_fwd:.2f}m fwd  y={y_side:.2f}m side")
            else:
                self.get_logger().info(f"LIMO id:{tid}  no valid depth")
 
 
def main():
    rclpy.init()
    node = MetricProjectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
 
 
if __name__ == "__main__":
    main()
 
#Note: Have to check if the pure Driver topic and the RGB (pixel) topic is aligned - meaning does they share the same resolution?
#ros2 topic echo /camera/color/image_raw --field width --field height --once
#ros2 topic echo /camera/depth/image_raw --field width --field height --once
#Also check:
#ros2 topic list | grep -iE "depth|aligned|registered|to_color", they might already have a depth aligned to color