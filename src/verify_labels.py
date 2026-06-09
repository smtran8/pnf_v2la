import argparse
import os
import glob
import cv2
"""
Check if the 

Reads images and their matching YOLO .txt label files, draws the boxes back
on, and shows them. 

This file should warn about:
  - coordinates outside (0,1)
  - a missing label file (image has no annotations at all)
  - unexpected class ids (should all be 0 for a single 'limo' class)

Usage:
    # Point at a folder of images (labels found automatically alongside):
    python verify_labels.py --images data-path-image

    # Explicit labels folder if it's separate from images:
    python verify_labels.py --images data-path-image --labels data-path-label


Controls:
    any key = next image,   q = quit
"""


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True, help="folder of images")
    p.add_argument("--labels", default=None,
                   help="folder of .txt labels")
    p.add_argument("--names", default="limo", help="comma-separated class names, in id order")
    p.add_argument("--no-show", action="store_true", help="validate only, no display")
    return p.parse_args()


def find_label_path(img_path, labels_dir):
    """Find the .txt that matches an image."""
    stem = os.path.splitext(os.path.basename(img_path))[0]
    if labels_dir:
        return os.path.join(labels_dir, stem + ".txt")
    # try same folder
    same = os.path.join(os.path.dirname(img_path), stem + ".txt")
    if os.path.exists(same):
        return same
    guess = img_path
    for img_seg, lbl_seg in (("images", "labels"),):
        guess = guess.replace(os.sep + img_seg + os.sep, os.sep + lbl_seg + os.sep)
    guess = os.path.splitext(guess)[0] + ".txt"
    return guess


def main():
    args = parse_args()
    names = [n.strip() for n in args.names.split(",")]

    images = []
    for ext in IMG_EXTS:
        images.extend(glob.glob(os.path.join(args.images, "*" + ext)))
    images.sort()

    if not images:
        print(f"No images found in {args.images}")
        return

    total_boxes = 0
    problems = 0

    for img_path in images:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Skip - could not read {img_path}")
            continue
        h, w = img.shape[:2]

        label_path = find_label_path(img_path, args.labels)
        base = os.path.basename(img_path)

        if not os.path.exists(label_path):
            print(f"[WARN] {base}: NO label file found ({label_path}), treated as background")
            problems += 1
            n_boxes = 0
        else:
            with open(label_path) as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            n_boxes = len(lines)
            for ln in lines:
                parts = ln.split()
                if len(parts) != 5:
                    print(f"[ERR ] {base}: line has {len(parts)} values, expected 5 -> '{ln}'")
                    problems += 1
                    continue
                cls_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:])

                # format checks
                if not all(0.0 <= v <= 1.0 for v in (cx, cy, bw, bh)):
                    print(f"Error {base}: Coordinates outside of range (0,1)) -> {parts[1:]}")
                    problems += 1
                if cls_id >= len(names):
                    print(f"Error {base}: class id {cls_id} but only {len(names)} class name(s)")
                    problems += 1

                # convert normalized center/size -> pixel corners for drawing
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                label = names[cls_id] if cls_id < len(names) else f"id{cls_id}"
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img, label, (x1, max(y1 - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        total_boxes += n_boxes
        print(f"{base}: {n_boxes} box(es)")

        if not args.no_show:
            cv2.imshow("verify labels (any key = next, q = quit)", img)
            if cv2.waitKey(0) & 0xFF == ord("q"):
                break

    if not args.no_show:
        cv2.destroyAllWindows()

    print("\nSummary")
    print(f"images:      {len(images)}")
    print(f"total boxes: {total_boxes}")
    print(f"problems:    {problems}")
    if problems == 0:
        print("Looks clean. Safe to train.")
    else:
        print("Fix the problems above before training.")


if __name__ == "__main__":
    main()