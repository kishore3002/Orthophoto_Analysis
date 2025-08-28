from ultralytics import YOLO

model = YOLO(r"../models/best.pt")  

results = model('D:/project/Dataset/test/images/tile_73_jpg.rf.9885d3dcda127773e5234933e511bf96.jpg', device=0, save=True)
