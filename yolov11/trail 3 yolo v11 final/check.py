from ultralytics import YOLO

model = YOLO("best.pt")  
print(model)  # Check if model loads properly
