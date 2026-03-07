import cv2
import easyocr
from ultralytics import YOLO
import re
import numpy as np

IMAGE_PATH = '../../data/anpr/test/test_car.jpg' 
MODEL_PATH = '../../models/anpr/best_yolo_plate.pt'

print("🚀 Loading YOLOv8 & EasyOCR...")
yolo_model = YOLO(MODEL_PATH)
reader = easyocr.Reader(['en'], gpu=True)

def get_best_ocr_read(plate_crop):
    padded_crop = cv2.copyMakeBorder(plate_crop, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    
    gray = cv2.cvtColor(padded_crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    
    v1 = resized
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    v2 = clahe.apply(resized)
    
    blur = cv2.GaussianBlur(resized, (5, 5), 0)
    _, v3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    best_text = ""
    best_conf = 0.0
    
    for img_version in [v1, v2, v3]:
        ocr_results = reader.readtext(
            img_version, 
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- ', 
            width_ths=1.5,
            mag_ratio=2,
            paragraph=True 
        )
        
        stitched_text = "".join([t for _, t in ocr_results])
        
        if not stitched_text:
             ocr_results = reader.readtext(img_version, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- ', width_ths=1.5, mag_ratio=2)
             stitched_text = "".join([t for _, t, _ in ocr_results])
             avg_prob = sum([p for _, _, p in ocr_results]) / len(ocr_results) if ocr_results else 0
        else:
             avg_prob = 1.0 
        
        if stitched_text and avg_prob >= best_conf:
            best_conf = avg_prob
            best_text = stitched_text
            
    return best_text, best_conf

def clean_and_format_plate(raw_text):
    """The Ultimate Sri Lankan Plate Formatter"""
    clean_text = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
    
    if len(clean_text) < 5: 
        return None
        
    prefix = clean_text[:-4]
    suffix = clean_text[-4:]
    
    num_fixes = {'O':'0', 'I':'1', 'L':'1', 'S':'5', 'B':'8', 'Z':'2', 'A':'4', 'G':'6', 'Q':'0'}
    let_fixes = {'0':'O', '1':'I', '5':'S', '8':'B'}
    
    fixed_prefix = "".join([let_fixes.get(c, c) for c in prefix])
    fixed_suffix = "".join([num_fixes.get(c, c) for c in suffix])
    
    if not re.match(r'^\d{4}$', fixed_suffix):
        return None 
        
    fixed_prefix = re.sub(r'[^A-Z]', '', fixed_prefix)
    
    provinces = ['WP', 'CP', 'NW', 'NC', 'SP', 'UP', 'SG', 'EP', 'NP']
    
    for prov in provinces:
        if fixed_prefix.startswith(prov):
            fixed_prefix = fixed_prefix[2:]
            break
            
    if len(fixed_prefix) == 3 and fixed_prefix[0] in ['W', 'C', 'N', 'S', 'U', 'E']:
        fixed_prefix = fixed_prefix[1:] 

    if len(fixed_prefix) in [2, 3]:
        final_letters = fixed_prefix
    else:
        final_letters = fixed_prefix[-3:] if len(fixed_prefix) >= 3 else fixed_prefix

    return f"{final_letters}-{fixed_suffix}"

print(f"📸 Processing {IMAGE_PATH}...")
frame = cv2.imread(IMAGE_PATH)

if frame is None:
    print(f"❌ Error: Could not find '{IMAGE_PATH}'.")
    exit()

results = yolo_model(frame, verbose=False)[0]

plates_found = 0
for box in results.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    
    margin_x = int((x2 - x1) * 0.05)
    margin_y = int((y2 - y1) * 0.05)
    
    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(frame.shape[1], x2 + margin_x)
    y2 = min(frame.shape[0], y2 + margin_y)
    
    plate_crop = frame[y1:y2, x1:x2]
    raw_text, confidence = get_best_ocr_read(plate_crop)
    
    if raw_text:
        plates_found += 1
        valid_plate = clean_and_format_plate(raw_text)
        final_text = valid_plate if valid_plate else f"? {raw_text}"
        
        print(f"✅ Read {plates_found}: {final_text}")
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
        cv2.putText(frame, final_text, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

if plates_found == 0:
    print("⚠️ YOLO found a plate, but EasyOCR couldn't read the letters.")

cv2.imshow("IntelliPark Universal Plate Reader", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()