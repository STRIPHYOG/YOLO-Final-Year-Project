import cv2
from ultralytics import YOLO
import torch

# Load YOLO model and move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("best.pt").to(device)
names = model.names

# Open video file or webcam
cap = cv2.VideoCapture("LL9.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (1040, 640))  # Optimize frame size
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for better performance

    # Run YOLO detection
    results = model.track(frame_rgb, persist=True, conf=0.3)

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        confidences = results[0].boxes.conf.cpu().tolist()

        for box, class_id, track_id, conf in zip(boxes, class_ids, track_ids, confidences):
            x1, y1, x2, y2 = box
            label = f"{names[class_id]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("FRAME", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
