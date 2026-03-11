from ultralytics import YOLO
import cv2
import math
import cvzone
import numpy as np

# ==========================================
# 1. Configuration & Model Loading
# ==========================================
# Make sure these point to your downloaded .pt files
plate_model = YOLO("plate_model.pt")
letter_model = YOLO("yolov8_char_model.pt")

# The exact class mapping to fix the index offset in your dataset
# ==========================================
# 2. Character Reading Logic
# ==========================================
def read_plate(char_model, crop_img_bgr):
    """Detects and sorts characters from a BGR cropped plate image."""
    results = char_model(crop_img_bgr, imgsz=320, conf=0.25, verbose=False)
    
    if len(results[0].boxes) == 0:
        return "Reading..."
        
    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    names = letter_model.names

    chars = []
    for box, cls in zip(boxes, classes):
        x1, y1, x2, y2 = box
        char_str = names[int(cls)]
        chars.append({'x': x1, 'y': y1, 'char': char_str, 'h': y2 - y1})

    if not chars:
        return "Reading..."

    # Sort top-to-bottom first
    chars.sort(key=lambda item: item['y'])
    avg_h = sum(c['h'] for c in chars) / max(len(chars), 1)

    # Group into lines dynamically
    lines = [[chars[0]]]
    for i in range(1, len(chars)):
        if chars[i]['y'] - lines[-1][-1]['y'] > 0.4 * avg_h:
            lines.append([chars[i]])
        else:
            lines[-1].append(chars[i])

    # Sort left-to-right and combine
    final_text = ""
    for line in lines:
        line.sort(key=lambda item: item['x'])
        final_text += "".join([c['char'] for c in line])

    return final_text

# ==========================================
# 3. Main Webcam Loop
# ==========================================
# Change to 0 for built-in webcam, or keep 1 for external USB webcam
cap = cv2.VideoCapture(1) 
cap.set(3, 1280)
cap.set(4, 720)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to grab frame. Check your camera index.")
        break
        
    # Stage 1: Detect Plate
    results = plate_model(img, stream=True, conf=0.5)
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get Bounding Box Coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            w, h = x2 - x1, y2 - y1
            
            # Get Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            
            # Draw the stylish corner rectangle using cvzone
            cvzone.cornerRect(img, (x1, y1, w, h), l=15, t=3, rt=1, colorR=(0, 255, 0), colorC=(0, 255, 0))
            
            # --- Stage 2: Read Plate Characters ---
            img_h, img_w, _ = img.shape
            px1, py1 = max(0, x1 - 2), max(0, y1 - 2)
            px2, py2 = min(img_w, x2 + 2), min(img_h, y2 + 2)
            
            crop_bgr = img[py1:py2, px1:px2]
            crop_h, crop_w = crop_bgr.shape[:2]
            
            plate_text = "Reading..."
            
            # Ensure crop is valid and upscale if it's too small
            if crop_w > 0 and crop_h > 0:
                if crop_w < 150:
                    crop_bgr = cv2.resize(crop_bgr, (crop_w * 3, crop_h * 3), interpolation=cv2.INTER_CUBIC)
                
                plate_text = read_plate(letter_model, crop_bgr)
            
            # Display the result using cvzone's styled text box
            cvzone.putTextRect(img, f'{plate_text} ({conf})', (max(0, x1), max(35, y1 - 10)), 
                               scale=1.5, thickness=2, colorR=(0, 200, 0))

    cv2.imshow("ALPR Live Feed", img)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()