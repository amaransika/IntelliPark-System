import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import re
import pandas as pd
from datetime import datetime
import os
import time

VIDEO_SOURCE = '../../data/anpr/manual_test_video_4.mp4' 

print("Loading YOLOv8 Plate Detector...")
yolo_model = YOLO('../../models/anpr/best_yolo_plate.pt')

print("Loading EasyOCR Engine...")
reader = easyocr.Reader(['en'], gpu=True) 

FLAGGED_CSV_PATH = '../../data/anpr/flagged_vehicles.csv'
LOG_CSV_PATH = '../..data/anpr/anpr_log.csv'

if not os.path.exists(FLAGGED_CSV_PATH):
    pd.DataFrame(columns=['plate']).to_csv(FLAGGED_CSV_PATH, index=False)
if not os.path.exists(LOG_CSV_PATH):
    pd.DataFrame(columns=['Date', 'Time', 'Plate', 'Status']).to_csv(LOG_CSV_PATH, index=False)

flagged_df = pd.read_csv(FLAGGED_CSV_PATH)
GLOBAL_FLAGGED_LIST = [str(p).strip().upper() for p in flagged_df['plate'].tolist()]
recently_logged_plates = {}
LOG_COOLDOWN_SECONDS = 10 


def get_best_ocr_read(plate_crop):
    padded_crop = cv2.copyMakeBorder(plate_crop, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    
    gray = cv2.cvtColor(padded_crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    
    v1 = resized
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    v2 = clahe.apply(resized)
    blur = cv2.GaussianBlur(resized, (5, 5), 0)
    _, v3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel = np.ones((3,3), np.uint8)
    v4_dilated = cv2.erode(v3, kernel, iterations=1) 
    
    best_text = ""
    best_conf = 0.0
    
    for img_version in [v1, v2, v3, v4_dilated]:
        ocr_results = reader.readtext(
            img_version, 
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- ', 
            width_ths=1.5,
            mag_ratio=2,
            contrast_ths=0.05,
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
        if prov in fixed_prefix:
            fixed_prefix = fixed_prefix.replace(prov, '') 
            
    if len(fixed_prefix) > 3 and fixed_prefix[0] in ['W', 'C', 'N', 'S', 'U', 'E']:
        fixed_prefix = fixed_prefix[1:] 

    if len(fixed_prefix) >= 3:
        final_letters = fixed_prefix[-3:] 
    elif len(fixed_prefix) == 2:
        final_letters = fixed_prefix 
    else:
        return None 

    return f"{final_letters}-{fixed_suffix}"

def log_plate(plate_text):
    current_time = time.time()
    if plate_text in recently_logged_plates:
        if current_time - recently_logged_plates[plate_text] < LOG_COOLDOWN_SECONDS:
            return 
            
    is_flagged = plate_text in GLOBAL_FLAGGED_LIST
    status = "FLAGGED ALARM" if is_flagged else "Authorized"
    now = datetime.now()
    
    pd.DataFrame([{
        'Date': now.strftime("%Y-%m-%d"),
        'Time': now.strftime("%H:%M:%S"),
        'Plate': plate_text,
        'Status': status
    }]).to_csv(LOG_CSV_PATH, mode='a', header=False, index=False)
    
    recently_logged_plates[plate_text] = current_time
    print("-" * 50)
    if is_flagged:
        print(f"SECURITY ALERT: Flagged vehicle {plate_text} detected at {now.strftime('%H:%M:%S')}!")
    else:
        print(f"LOGGED: {plate_text} granted entry.")
    print("-" * 50)


cap = cv2.VideoCapture(VIDEO_SOURCE)
frame_count = 0

active_session = False
best_plate_text = None
best_plate_conf = 0.0
frames_without_detection = 0
SESSION_TIMEOUT = 10 

print("\n ANPR System Live. Waiting for vehicles...\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
        
    frame_count += 1
    if frame_count % 3 != 0:
        cv2.imshow("IntelliPark ANPR Security Gateway", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        continue

    results = yolo_model(frame, verbose=False)[0]
    saw_plate_this_frame = False
    is_currently_flagged = best_plate_text in GLOBAL_FLAGGED_LIST if best_plate_text else False
    
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        w, h = x2 - x1, y2 - y1
        
        if w > 50 and h > 15:
            saw_plate_this_frame = True
            active_session = True
            frames_without_detection = 0 
            
            aspect_ratio = w / float(h)
            if aspect_ratio > 1.8:
                shave_amount = int(w * 0.16)
                x1 = x1 + shave_amount
            
            margin_x = int((x2 - x1) * 0.05)
            margin_y = int((y2 - y1) * 0.05)
            x1_crop = max(0, x1 - margin_x)
            y1_crop = max(0, y1 - margin_y)
            x2_crop = min(frame.shape[1], x2 + margin_x)
            y2_crop = min(frame.shape[0], y2 + margin_y)
            
            plate_crop = frame[y1_crop:y2_crop, x1_crop:x2_crop]
            
            try:
                raw_text, avg_prob = get_best_ocr_read(plate_crop)
                
                if raw_text and avg_prob > best_plate_conf: 
                    valid_plate = clean_and_format_plate(raw_text)
                    if valid_plate:
                        best_plate_text = valid_plate
                        best_plate_conf = avg_prob
                        is_currently_flagged = best_plate_text in GLOBAL_FLAGGED_LIST
                
                box_color = (0, 0, 255) if is_currently_flagged else (0, 255, 0)
                draw_x1, draw_y1, draw_x2, draw_y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (draw_x1, draw_y1), (draw_x2, draw_y2), box_color, 3)
                
                if best_plate_text:
                    status_text = "FLAGGED" if is_currently_flagged else "AUTHORIZED"
                    display_text = f"{best_plate_text} [{status_text}]"
                    cv2.putText(frame, display_text, (draw_x1, draw_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)
            
            except Exception as e:
                pass 

    if active_session and is_currently_flagged:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (0, 0, 255), -1)
        cv2.putText(frame, f"SECURITY ALERT: FLAGGED VEHICLE {best_plate_text}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    if active_session and not saw_plate_this_frame:
        frames_without_detection += 1
        if frames_without_detection > SESSION_TIMEOUT:
            if best_plate_text:
                log_plate(best_plate_text)
            
            active_session = False
            best_plate_text = None
            best_plate_conf = 0.0
            frames_without_detection = 0

    cv2.imshow("IntelliPark ANPR Security Gateway", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()