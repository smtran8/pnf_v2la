import argparse
import cv2
from ultralytics import YOLO

"""Build a Tracker system that puts an ID on every LIMO agent.
- Method: Run Detector+ Reuse ByteTrack algorithm. Source: https://arxiv.org/abs/2110.06864
- Usage:
# Test on a video file:
python tracker.py --source path/to/video
# Test on webcam:
python tracker.py --source 0

q = quit

#Notes:
- Always set persist= True, keeping track of the model through time
- Reuse best.pt
"""

# Different colors per track id
COLORS = [
    (0, 0, 255), (0, 200, 0), (255, 0, 0), (0, 200, 200),
    (200, 0, 200), (0, 140, 255), (200, 200, 0), (128, 0, 255),
]
 
 
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="0",
                   help="'0' for webcam, or path to a video file")
    p.add_argument("--model", default="best.pt",
                   help="path to your trained LIMO weights")
    p.add_argument("--conf", type=float, default=0.3,
                   help="confidence threshold (matches your F1 sweet spot)")
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model)
    source = int(args.source) if args.source.isdigit() else args.source
 
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: could not open source '{args.source}'")
        return
 
    print("Running tracker. Press 'q' to quit.")
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            print("End of stream.")
            break
 
        # Call: Detect + ByteTrack together.
        # persist=True keeps tracks alive across frames (essential).
        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=args.conf,
            verbose=False,
        )
 
        ids_this_frame = []
 
        # Recall xyxy - x1y1 is the top left corner, x2y2 is the bottom right corner
        for box in results[0].boxes:
            # box.id is None until a track is confirmed, so guard per box
            if box.id is None:
                continue
            tid = int(box.id[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
 
            color = COLORS[tid % len(COLORS)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"LIMO id:{tid} {conf:.2f}",
                        (x1, max(y1 - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            ids_this_frame.append(tid)
 
        print(f"[frame {frame_idx}] active track IDs: {sorted(ids_this_frame)}")
 
        cv2.imshow("P&F tracker test (press 'q' to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        frame_idx += 1
 
    cap.release()
    cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()