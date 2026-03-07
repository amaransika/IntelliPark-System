# backend/main.py

import os
import cv2
import re
import numpy as np
import asyncio
import pandas as pd
import time
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List

# NEW: Import Supabase
from supabase import create_client, Client

# --- 1. Data Structures (CNN Parking) ---
class Coordinate(BaseModel):
    x: int
    y: int
    w: int
    h: int

class PredictionRequest(BaseModel):
    image_path: str
    weather: str
    coordinates: List[Coordinate]

class PredictionResponse(BaseModel):
    is_busy: bool
    confidence: float

# --- 2. Initialize FastAPI ---
app = FastAPI(title="IntelliPark Master AI Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPABASE_URL = "https://jzxlafustfcreyxjulvq.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imp6eGxhZnVzdGZjcmV5eGp1bHZxIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzI4NTkwMTYsImV4cCI6MjA4ODQzNTAxNn0.2iKD2qjXuQ8UxgU91C5S-La8TajF2_9sk7WROKAfWZo"

# Initialize Supabase client (only if keys are provided)
supabase: Client = None
if "https://jzxlafustfcreyxjulvq.supabase.co" not in SUPABASE_URL:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("☁️ [INFO] Supabase Cloud Database Connected Successfully!")
else:
    print("⚠️ [WARNING] Supabase Keys not added. Cloud saving will be disabled.")


# Global Variables (CNN)
parking_model = None
lb = None

# Global Variables (ANPR)
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANPR_VIDEO_PATH = os.path.join(base_dir, 'data', 'anpr', 'manual_test_video_4.mp4')
FLAGGED_CSV_PATH = os.path.join(base_dir, 'data', 'anpr', 'flagged_vehicles.csv')
LOG_CSV_PATH = os.path.join(base_dir, 'data', 'anpr', 'anpr_log.csv')
YOLO_PATH = os.path.join(base_dir, 'models', 'anpr', 'best_yolo_plate.pt')

GLOBAL_ANPR_FRAME = None
ANPR_STATE = {
    "best_plate": "--",
    "confidence": 0.0,
    "is_flagged": False,
    "alert_message": "System Initializing..."
}
PLATE_HISTORY = []
ANPR_IS_RUNNING = False 

# Pre-loaded Models
yolo_model_global = None
reader_global = None

# --- 3. Asynchronous Delayed Pre-Loader (Super Fast Startup) ---
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(delayed_ai_loader())

async def delayed_ai_loader():
    global parking_model, lb, yolo_model_global, reader_global
    
    print("⏳ [INFO] Waiting 5 seconds for Uvicorn to bind to port...")
    await asyncio.sleep(5)
    
    # 1. Load Parking CNN
    try:
        ANPR_STATE["alert_message"] = "Loading Parking CNN..."
        print("⚙️ [INFO] Loading Parking CNN in background...")
        def load_cnn():
            import tensorflow as tf
            from sklearn.preprocessing import LabelBinarizer
            model_path = os.path.join(base_dir, 'models', 'best_parking_model.keras')
            model = tf.keras.models.load_model(model_path)
            lb_instance = LabelBinarizer()
            lb_instance.fit(['OVERCAST', 'RAINY', 'SUNNY'])
            return model, lb_instance
        parking_model, lb = await asyncio.to_thread(load_cnn)
        print("✅ [INFO] Multi-Modal Parking CNN Loaded.")
    except Exception as e:
        print(f"❌ [ERROR] CNN Loading failed: {e}")

    # 2. Pre-load YOLO & EasyOCR in background (Ready for button click)
    try:
        ANPR_STATE["alert_message"] = "Pre-loading ANPR AI Engines..."
        print("⚙️ [INFO] Pre-loading YOLO & EasyOCR in background...")
        def load_anpr():
            from ultralytics import YOLO
            import easyocr
            return YOLO(YOLO_PATH), easyocr.Reader(['en'], gpu=False)
        yolo_model_global, reader_global = await asyncio.to_thread(load_anpr)
        print("✅ [INFO] ANPR Models Pre-loaded Successfully.")
        ANPR_STATE["alert_message"] = "System Ready. Awaiting ANPR Activation..."
    except Exception as e:
        print(f"❌ [ERROR] ANPR Pre-loading failed: {e}")
        ANPR_STATE["alert_message"] = "Error Pre-loading AI Models!"

# --- 4. ANPR Core Logic ---
def clean_and_format_plate(raw_text):
    clean_text = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
    if len(clean_text) >= 6:
        prefix, suffix = clean_text[:-4], clean_text[-4:]
        num_fixes = {'O':'0', 'I':'1', 'L':'1', 'S':'5', 'B':'8', 'Z':'2', 'A':'4', 'G':'6', 'Q':'0'}
        let_fixes = {'0':'O', '1':'I', '5':'S', '8':'B'}
        clean_text = "".join([let_fixes.get(c, c) for c in prefix]) + "".join([num_fixes.get(c, c) for c in suffix])
        
    match = re.search(r'([A-Z]+)(\d{4})$', clean_text)
    if match:
        letters, numbers = match.group(1), match.group(2)
        if len(letters) == 5: final_letters = letters[2:]   
        elif len(letters) == 4: final_letters = letters[2:] 
        elif len(letters) == 3: final_letters = letters     
        elif len(letters) == 2: final_letters = letters     
        else: final_letters = letters[-2:] 
        return f"{final_letters}-{numbers}"
    return None

# --- NEW: Cloud DB Saver Helper Function ---
def save_to_cloud_db(date_str, time_str, plate, status):
    """Saves to Supabase Database securely without crashing the system"""
    if supabase is None:
        return
    try:
        data = {
            "date": date_str,
            "time": time_str,
            "plate": plate,
            "status": status
        }
        response = supabase.table('anpr_logs').insert(data).execute()
        print(f"☁️ [CLOUD DB] Synced securely: {plate} at {time_str}")
    except Exception as e:
        print(f"❌ [CLOUD DB ERROR] Failed to sync {plate}: {e}")

async def run_anpr_sentinel():
    global GLOBAL_ANPR_FRAME, ANPR_STATE, PLATE_HISTORY, ANPR_IS_RUNNING, yolo_model_global, reader_global
    
    if not os.path.exists(YOLO_PATH):
        ANPR_STATE["alert_message"] = "Error: YOLO Model Not Found!"
        ANPR_IS_RUNNING = False
        return

    while yolo_model_global is None or reader_global is None:
        ANPR_STATE["alert_message"] = "AI Models still warming up, please wait..."
        await asyncio.sleep(1)

    yolo_model = yolo_model_global
    reader = reader_global
    
    ANPR_STATE["alert_message"] = "Scanning Entrance..."
    
    if not os.path.exists(FLAGGED_CSV_PATH): pd.DataFrame(columns=['plate']).to_csv(FLAGGED_CSV_PATH, index=False)
    if not os.path.exists(LOG_CSV_PATH): pd.DataFrame(columns=['Date', 'Time', 'Plate', 'Status']).to_csv(LOG_CSV_PATH, index=False)

    flagged_df = pd.read_csv(FLAGGED_CSV_PATH)
    global_flagged_list = [str(p).strip().upper() for p in flagged_df['plate'].tolist()]

    cap = cv2.VideoCapture(ANPR_VIDEO_PATH)
    frame_count, active_session, best_plate_text, best_plate_conf = 0, False, None, 0.0
    frames_without_detection, SESSION_TIMEOUT = 0, 10 
    recently_logged_plates = {}
    is_currently_flagged = False 

    while ANPR_IS_RUNNING:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
            
        frame_count += 1
        if frame_count % 3 != 0:
            await asyncio.sleep(0.01)
            continue

        results = await asyncio.to_thread(yolo_model, frame, verbose=False)
        boxes = results[0].boxes
        saw_plate_this_frame = False
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            
            if w > 50 and h > 15:
                saw_plate_this_frame, active_session, frames_without_detection = True, True, 0 
                plate_crop = frame[y1:y2, x1:x2]
                
                try:
                    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    processed_plate = cv2.resize(clahe.apply(gray), None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
                    
                    ocr_results = await asyncio.to_thread(reader.readtext, processed_plate, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                    stitched_text = "".join([t for _, t, _ in ocr_results])
                    avg_prob = sum([p for _, _, p in ocr_results]) / len(ocr_results) if ocr_results else 0
                    
                    if stitched_text and avg_prob > best_plate_conf and avg_prob > 0.15: 
                        valid_plate = clean_and_format_plate(stitched_text)
                        if valid_plate and avg_prob > 0.90:
                            best_plate_text, best_plate_conf = valid_plate, avg_prob
                            is_currently_flagged = best_plate_text in global_flagged_list
                            
                            ANPR_STATE.update({
                                "best_plate": best_plate_text,
                                "confidence": round(best_plate_conf, 2),
                                "is_flagged": is_currently_flagged,
                                "alert_message": f"🚨 ALERT: {best_plate_text}" if is_currently_flagged else f"✅ Scanning: {best_plate_text}"
                            })

                    if best_plate_text:
                        box_color = (0, 0, 255) if is_currently_flagged else (0, 255, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
                        status_text = "FLAGGED" if is_currently_flagged else "AUTHORIZED"
                        cv2.putText(frame, f"{best_plate_text} [{status_text}]", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)
                
                except Exception: pass 

        if active_session and is_currently_flagged:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (0, 0, 255), -1)
            cv2.putText(frame, f"SECURITY ALERT: FLAGGED VEHICLE {best_plate_text}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        if active_session and not saw_plate_this_frame:
            frames_without_detection += 1
            if frames_without_detection > SESSION_TIMEOUT:
                if best_plate_text:
                    curr_time = time.time()
                    if best_plate_text not in recently_logged_plates or (curr_time - recently_logged_plates[best_plate_text] > 10):
                        is_flagged = best_plate_text in global_flagged_list
                        now = datetime.now()
                        date_str = now.strftime("%Y-%m-%d")
                        time_str = now.strftime("%H:%M:%S")
                        status_str = "🚨 FLAGGED ALARM" if is_flagged else "✅ Authorized"
                        
                        # 1. Save to Local CSV (Original Logic)
                        pd.DataFrame([{
                            'Date': date_str, 'Time': time_str,
                            'Plate': best_plate_text, 'Status': status_str
                        }]).to_csv(LOG_CSV_PATH, mode='a', header=False, index=False)
                        
                        # 2. Add to Frontend UI
                        PLATE_HISTORY.insert(0, {"plate": best_plate_text, "flagged": is_flagged, "time": time_str})
                        if len(PLATE_HISTORY) > 6: PLATE_HISTORY.pop()
                        
                        # 3. Save to Real-World Cloud Database ASYNCHRONOUSLY (No Lag)
                        asyncio.create_task(asyncio.to_thread(save_to_cloud_db, date_str, time_str, best_plate_text, status_str))
                        
                        recently_logged_plates[best_plate_text] = curr_time

                active_session, best_plate_text, best_plate_conf, frames_without_detection, is_currently_flagged = False, None, 0.0, 0, False
                ANPR_STATE.update({"best_plate": "--", "confidence": 0.0, "is_flagged": False, "alert_message": "Scanning Entrance..."})

        GLOBAL_ANPR_FRAME = frame.copy()
        await asyncio.sleep(0.01)

# --- 5. API Endpoints ---
@app.get("/api/anpr/start")
async def start_anpr_engine():
    global ANPR_IS_RUNNING, ANPR_STATE
    if not ANPR_IS_RUNNING:
        ANPR_IS_RUNNING = True
        ANPR_STATE["alert_message"] = "Connecting Camera..."
        asyncio.create_task(run_anpr_sentinel())
        return {"status": "ANPR Engine Started successfully"}
    return {"status": "ANPR Engine is already running"}

@app.post("/predict", response_model=List[PredictionResponse])
def predict_occupancy(req: PredictionRequest):
    if parking_model is None:
        raise HTTPException(status_code=503, detail="Model is still loading. Please wait.")
        
    full_img_path = os.path.join(base_dir, req.image_path)
    img = cv2.imread(full_img_path)
    if img is None: raise HTTPException(status_code=404, detail="Image not found")
    
    h, w, _ = img.shape
    w_vec = lb.transform([req.weather])[0].astype('float32')
    w_batch = np.tile(w_vec, (len(req.coordinates), 1))
    
    patches = []
    for coord in req.coordinates:
        x1, y1, x2, y2 = max(0, coord.x), max(0, coord.y), min(w, coord.x + coord.w), min(h, coord.y + coord.h)
        patch = img[y1:y2, x1:x2]
        patch = cv2.resize(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB), (150, 150)) if patch.size > 0 else np.zeros((150, 150, 3))
        patches.append(patch / 255.0)
        
    preds = parking_model.predict([np.array(patches), w_batch], verbose=0)
    return [{"is_busy": float(conf[0]) > 0.5, "confidence": float(conf[0])} for conf in preds]

@app.get("/api/anpr/video_feed")
async def anpr_video_feed():
    async def stream_gen():
        while True:
            if GLOBAL_ANPR_FRAME is not None:
                ret, buffer = cv2.imencode('.jpg', GLOBAL_ANPR_FRAME)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            else:
                blank = np.zeros((480, 640, 3), np.uint8)
                msg = ANPR_STATE.get("alert_message", "Standby...")
                cv2.putText(blank, msg, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                ret, buffer = cv2.imencode('.jpg', blank)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            await asyncio.sleep(0.05)
    return StreamingResponse(stream_gen(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/anpr/status")
async def get_anpr_status():
    return {"feed": ANPR_STATE, "history": PLATE_HISTORY}

@app.api_route("/", methods=["GET", "HEAD"])
def read_root():
    return {"status": "Active", "message": "IntelliPark API is running."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)