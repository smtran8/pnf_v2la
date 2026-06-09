import argparse
import cv2 
from ultralytics import YOLO

"""Goal: find out whether a stock COCO-pretrained YOLO already recognizes a LIMO
(likely as "car", COCO class 2)
Usage:
    # Live webcam (default camera 0):
    python detector.py

    # A recorded video file:
    python detector.py --source path/to/video.mp4

    # A single image:
    python detector.py --source path/to/limo.jpg

    # Lower the confidence threshold to see uncertainty:
    python detector.py --conf 0.15. Default is 0.25. If we lower 0.15 and see a LIMO car label, but disappear at 0.25, that is weak detection

Controls (live/video window):
    q  = quit"""
    
    
def parse_args():
    p = argparse.ArgumentParser(description="YOLO LIMO detection test")
    p.add_argument("--source", default="0",
                   help="'0' for webcam, or a path to an image/video file")
    p.add_argument("--model", default="yolo26n.pt",
                   help="YOLO weights (e.g. yolo.26n.pt, yolo11n.pt). Auto-downloads on first run")
    p.add_argument("--conf", type=float, default=0.25,
                   help="Confidence threshold (lower = more, weaker detections)")
    return p.parse_args()

def is_image(path: str) -> bool:
    return path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))

def draw_and_report(frame, result, model_names):
    """Draw boxes, print a summary of detected classes + confidences every frame"""
    detections = []
    # result.boxes holds 2 corners that pin down the car: x1,y1 and x2,y2 coords, confidences, and class indices
    for box in result.boxes:
        cls_id = int(box.cls[0])
        label = model_names[cls_id]
        ALLOWED = {"limo"}

        
        #if label not in ALLOWED:
            #continue
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        detections.append((label, conf))
 
        # Highlight 'car' (COCO class 2) in a different color (RED) since we assume a LIMO will be considered a car
        color = (0, 0, 255) 
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
 
    if detections:
        summary = ", ".join(f"{lbl}({c:.2f})" for lbl, c in detections)
        print(f"  detected: {summary}")
    else:
        print("  detected: (no car/truck/bus above threshold)")
 
    return frame


def main():
    args = parse_args()
    print(f"Loading model: {args.model}  (downloads automatically on first use)")
    model = YOLO(args.model)
    names = model.names  # dict: class_id -> class_name
 
    # For Image
    if is_image(args.source):
        results = model(args.source, conf=args.conf)
        frame = results[0].orig_img.copy()
        print(f"Image: {args.source}")
        frame = draw_and_report(frame, results[0], names)
        cv2.imshow("P&F detector test (press any key to close)", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return
 
    # For Webcam/Video
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: could not open source '{args.source}'")
        return
 
    print("Running. Press 'q' in the window to quit.")
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            print("End of stream.")
            break
 
        # verbose=False so YOLO doesn't spam its own per-frame log
        results = model(frame, conf=args.conf, verbose=False)
        print(f"[frame {frame_idx}]", end="")
        frame = draw_and_report(frame, results[0], names)
 
        cv2.imshow("P&F detector test (press 'q' to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        frame_idx += 1
 
    cap.release()
    cv2.destroyAllWindows()


 
if __name__ == "__main__":
    main()