"""
This file is to test YOLO on the first 26  images to ensure the model is run correctly before collecting more data/fine-tuning
"""

from ultralytics import YOLO


def main():
    # Transfer Learning - Start from the small COCO-pretrained checkpoint 
    model = YOLO("yolo26n.pt")  

    model.train(
        data="data.yaml",   # 
        epochs=10,                 # Test - small
        imgsz=640,
        batch=4,                   # small batch, only 25 image
        device="cpu",              
        #project="runs/detect",
        name="limo_dryrun",
        verbose=True,
    )

    print("\n=== Dry-run complete ===")
    print("best.pt outputted in the directory: runs/detect/limo_dryrun/weights/,")
    print("There are only 25 images. Next step: collect the full dataset and train on HiPerGator.")


if __name__ == "__main__":
    main()