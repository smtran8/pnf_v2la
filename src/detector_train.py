"""
This file is to test YOLO on the first 26  images to ensure the model is run correctly before collecting more data/fine-tuning
"""

from ultralytics import YOLO


def main():
    # Transfer Learning - Start from the small COCO-pretrained checkpoint 
    model = YOLO("yolo26n.pt")  

    model.train(
    data="data.yaml",
    epochs=100,         
    imgsz=640,
    batch=8,            
    device='cpu',            # '0' for gpu
    patience=20,         # Stop early if val stops improving for 20 epochs (prevents overfit + wasted time)
    #project="runs/detect",
    name="limo_full",    # 
    verbose=True,
)

    print("\n=== Dry-run complete ===")
    print("best.pt outputted in the directory: runs/detect")
    print("There are around 200 images now. Note that image could be duplicates, so the result is pretty optimistic")


if __name__ == "__main__":
    main()