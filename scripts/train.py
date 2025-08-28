# from ultralytics import YOLO

# if __name__ == "__main__":
#     model = YOLO('models/yolov8n-seg.pt')  
#     model.train(
#         data='data.yaml',
#         epochs=100,
#         imgsz=640,
#         batch=8,
#         device=0  # Use GPU
#     )
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"../models/yolov8n-seg.pt")  # Load pretrained YOLOv8 segmentation model
    
    model.train(
        data="data.yaml",     # Path to dataset config (classes, paths to train/val/test)
        epochs=150,           # Train for 150 epochs (more epochs → better accuracy)
        imgsz=640,            # Input image size (match your dataset size)
        batch=8,              # Number of images per batch (adjust if GPU memory low)
        device=0,             # GPU id (0 for first GPU, "cpu" for CPU training)
        
        # Training optimization
        lr0=0.001,            # Initial learning rate
        optimizer="AdamW",    # Optimizer (AdamW better for segmentation)
        patience=30,          # Early stopping if no improvement in 30 epochs
        
        # Data augmentations (improves generalization + precision)
        mosaic=1.0,           # Combines 4 images → stronger context
        mixup=0.2,            # Mix two images together → improves robustness
        copy_paste=0.3,       # Copy objects from one image to another
        hsv_h=0.015,          # Random Hue shift
        hsv_s=0.7,            # Random Saturation shift
        hsv_v=0.4,            # Random Brightness shift
        flipud=0.5,           # Vertical flip (50% chance)
        fliplr=0.5,           # Horizontal flip (50% chance)
        degrees=15,           # Random rotation (±15 degrees)
        scale=0.5,            # Random zoom in/out (50%)
        shear=0.2,            # Shear transform (tilting)
        perspective=0.001,    # Small perspective warp
        translate=0.2,        # Random shifts (20% of image)
        
        # Saving & monitoring
        save_period=10,       # Save model weights every 10 epochs
        plots=True            # Save training curves & results
    )




