from ultralytics import YOLO

def evaluate_model():
    
    model = YOLO(r"../models/best.pt")

    
    metrics = model.val(data="D:/project/data.yaml", split="val", imgsz=640)

    
    print("✅ Evaluation Results:")
    print(f"mAP@0.5: {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()  
    evaluate_model()


