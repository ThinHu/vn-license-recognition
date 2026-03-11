import cv2
from ultralytics import YOLO
from utils import read_plate # Assuming you modularized the helper function

# Load Models
plate_model = YOLO('models/plate_model.pt')
letter_model = YOLO('models/letter_model.pt')

# Read Image
img_bgr = cv2.imread('test_image.jpg')

# Stage 1: Detect Plate
plate_results = plate_model(img_bgr, imgsz=640, conf=0.6)

if len(plate_results[0].boxes) > 0:
    box = plate_results[0].boxes[0].xyxy[0].cpu().numpy().astype(int)
    x1, y1, x2, y2 = box
    
    # Crop and pad
    h, w, _ = img_bgr.shape
    crop_bgr = img_bgr[max(0, y1-2):min(h, y2+2), max(0, x1-2):min(w, x2+2)]
    
    # Stage 2: Read Characters
    plate_text = read_plate(letter_model, crop_bgr)
    print(f"Detected Plate: {plate_text}")
else:
    print("No plate detected.")