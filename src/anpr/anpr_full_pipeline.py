import cv2
import os
import csv
import time
import re
from ultralytics import YOLO
import easyocr

model_path = 'C:/Users/Dell/Documents/GitHub/IntelliPark_Project/models/best.pt' 
dataset_path = 'C:/Users/Dell/Documents/GitHub/IntelliPark_Project/data/sri_lanka_plates/' 
results_path = 'C:/Users/Dell/Documents/GitHub/IntelliPark_Project/src/results/detections/'
csv_log_path = 'C:/Users/Dell/Documents/GitHub/IntelliPark_Project/src/results/parking_log.csv'
watchlist_path = 'C:/Users/Dell/Documents/GitHub/IntelliPark_Project/data/flagged_vehicles.csv' 

os.makedirs(results_path, exist_ok=True)

model = YOLO(model_path)
reader = easyocr.Reader(['en'])

def load_watchlist():
    """Loads plate numbers to be flagged from a text file."""
    if os.path.exists(watchlist_path):
        with open(watchlist_path, 'r') as f:
            return [line.strip().upper() for line in f.readlines() if line.strip()]
    return []

def clean_plate_text(text):
    """Standardizes OCR text for consistent database comparison."""
    clean = re.sub(r'[^A-Z0-9]', '', text.upper())
    return clean

def process_sri_lankan_dataset():
    print(f"--- Starting IntelliPark ANPR Security Pipeline ---")
    
    flagged_plates = load_watchlist()
    print(f"Watchlist Loaded: {len(flagged_plates)} plates monitored.")

    total_images = 0
    total_detections = 0
    flagged_detections = 0

    with open(csv_log_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Timestamp', 'Filename', 'Clean_Plate', 'Confidence', 'Security_Status'])

    image_files = [f for f in os.listdir(dataset_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
    total_images = len(image_files)
    
    for img_name in image_files:
        img_path = os.path.join(dataset_path, img_name)
        img = cv2.imread(img_path)
        if img is None: continue
        
        results = model.predict(source=img, save=False, conf=0.5, verbose=False)
        
        for result in results:
            if len(result.boxes) > 0:
                total_detections += 1
            
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                
                ocr_result = reader.readtext(img[y1:y2, x1:x2])
                raw_text = ocr_result[0][-2] if ocr_result else ""
                clean_plate = clean_plate_text(raw_text)
                
                if clean_plate in flagged_plates and clean_plate != "":
                    status = "FLAGGED"
                    flagged_detections += 1
                    color = (0, 0, 255) 
                    print(f"⚠️  [ALERT] Flagged Vehicle: {clean_plate} in {img_name}")
                else:
                    status = "AUTHORIZED"
                    color = (0, 255, 0) 
                
                current_time = time.strftime('%Y-%m-%d %H:%M:%S')
                with open(csv_log_path, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([current_time, img_name, clean_plate, conf, status])
                
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, f"{clean_plate} ({status})", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv2.imwrite(os.path.join(results_path, f"security_check_{img_name}"), img)

    print("\n--- Final Security Performance Summary ---")
    print(f"Detection Rate:      {(total_detections/total_images)*100:.2f}%")
    print(f"Flagged Vehicles:    {flagged_detections}")
    print(f"Log saved to:        {csv_log_path}")

if __name__ == "__main__":
    process_sri_lankan_dataset()