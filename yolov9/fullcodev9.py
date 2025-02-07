import cv2
import numpy as np
from ultralytics import YOLO
import cvzone

# Load YOLOv9 model
model = YOLO("best.pt")
names = model.names

# Initial resolution (meters per pixel) - Adjust dynamically based on zoom
initial_resolution = 100  # Example: 100 meters per pixel
resolution_meters_per_pixel = initial_resolution

# Open the video file or webcam
cap = cv2.VideoCapture('LL9.mp4')
count = 0

# Initialize feature detector for scale estimation
orb = cv2.ORB_create()
ret, prev_frame = cap.read()
prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
keypoints_prev, descriptors_prev = orb.detectAndCompute(prev_frame_gray, None)

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    count += 1
    if count % 2 != 0:
        continue
    if not ret:
        break
    
    frame = cv2.resize(frame, (1020, 600))
    
    # Convert to grayscale for feature matching
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    keypoints_curr, descriptors_curr = orb.detectAndCompute(frame_gray, None)
    
    # Feature matching using BFMatcher
    if descriptors_prev is not None and descriptors_curr is not None:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors_prev, descriptors_curr)
        
        if len(matches) > 10:
            # Compute scale change using matched keypoint distances
            prev_pts = np.float32([keypoints_prev[m.queryIdx].pt for m in matches]).reshape(-1, 2)
            curr_pts = np.float32([keypoints_curr[m.trainIdx].pt for m in matches]).reshape(-1, 2)
            
            # Compute median keypoint displacement ratio
            distances_prev = np.linalg.norm(prev_pts - prev_pts.mean(axis=0), axis=1)
            distances_curr = np.linalg.norm(curr_pts - curr_pts.mean(axis=0), axis=1)
            scale_factor = np.median(distances_prev) / np.median(distances_curr)
            
            # Update resolution dynamically
            resolution_meters_per_pixel = initial_resolution * scale_factor
    
    # Run YOLOv11 tracking on the frame
    results = model.track(frame, persist=True)
    
    # Check if there are any boxes in the results
    if results[0].boxes is not None and results[0].boxes.id is not None:
        # Get the boxes, class IDs, track IDs, and confidences
        boxes = results[0].boxes.xyxy.int().cpu().tolist()  # Bounding boxes
        class_ids = results[0].boxes.cls.int().cpu().tolist()  # Class IDs
        track_ids = results[0].boxes.id.int().cpu().tolist()  # Track IDs
        confidences = results[0].boxes.conf.cpu().tolist()  # Confidence scores
        
        for box, class_id, track_id, conf in zip(boxes, class_ids, track_ids, confidences):
            c = names[class_id]
            x1, y1, x2, y2 = box
            bbox_width_pixels = x2 - x1  # Width of bounding box in pixels
            
            # Formula: Crater Diameter (km) = Bounding Box Width (pixels) * Resolution (meters/pixel) / 1000
            crater_size_km = (bbox_width_pixels * resolution_meters_per_pixel) / 1000  
            
            # Draw rectangle & display crater size
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cvzone.putTextRect(frame, f'{c} - {crater_size_km:.2f} km', (x1, y1), 1, 1)
    
    # Display the frame
    cv2.imshow("FRAME", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
    # Update previous frame data
    prev_frame_gray = frame_gray
    keypoints_prev, descriptors_prev = keypoints_curr, descriptors_curr

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()