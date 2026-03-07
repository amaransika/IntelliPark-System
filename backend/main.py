# backend/main.py

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import cv2
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from fastapi import Request


# --- 1. Define Data Structures (Pydantic Models) ---
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

# --- 2. Initialize FastAPI Application ---
app = FastAPI(title="IntelliPark Feature Fusion Engine")

# Global variables for Model and LabelBinarizer
model = None
lb = None

# --- 3. Startup Event: Load AI Model ---
@app.on_event("startup")
def load_assets():
    global model, lb
    try:
        # Construct path to the model relative to backend/main.py
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, 'models', 'best_parking_model.keras')
        
        model = tf.keras.models.load_model(model_path)
        
        weather_classes = ['OVERCAST', 'RAINY', 'SUNNY']
        lb = LabelBinarizer()
        lb.fit(weather_classes)
        print("✅ [INFO] Multi-Modal CNN Loaded Successfully.")
    except Exception as e:
        print(f"❌ [ERROR] Failed to load model: {e}")

# --- 4. Prediction Endpoint (The Core Logic) ---
@app.post("/predict", response_model=List[PredictionResponse])
def predict_occupancy(req: PredictionRequest):
    # Construct absolute path to the image
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_img_path = os.path.join(base_dir, req.image_path)
    
    if not os.path.exists(full_img_path):
        raise HTTPException(status_code=404, detail=f"Image not found at path: {full_img_path}")
        
    img = cv2.imread(full_img_path)
    if img is None:
         raise HTTPException(status_code=500, detail="Failed to read the image file.")
    h, w, _ = img.shape
    
    # Process environmental context (Weather)
    w_vec = lb.transform([req.weather])[0].astype('float32')
    w_batch = np.tile(w_vec, (len(req.coordinates), 1))
    
    patches = []
    
    # Process visual context (Image Patches)
    for coord in req.coordinates:
        # Ensure coordinates are within image boundaries
        x1, y1 = max(0, coord.x), max(0, coord.y)
        x2, y2 = min(w, coord.x + coord.w), min(h, coord.y + coord.h)
        
        patch = img[y1:y2, x1:x2]
        
        if patch.size > 0:
            patch = cv2.resize(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB), (150, 150))
        else:
            patch = np.zeros((150, 150, 3), dtype=np.uint8)
            
        patches.append(patch / 255.0)
        
    # Execute Model Inference (Feature Fusion occurs here)
    preds = model.predict([np.array(patches), w_batch], verbose=0)
    
    # Format the results
    results = []
    for conf in preds:
        confidence_val = float(conf[0])
        results.append(PredictionResponse(
            is_busy=confidence_val > 0.5,
            confidence=confidence_val
        ))
        
    return results

# --- 5. Health Check Endpoint ---
@app.get("/")
def read_root():
    return {"status": "Active", "message": "IntelliPark API is running."}


@app.api_route("/", methods=["GET", "HEAD"])
def read_root(request: Request):
    return {"status": "Active", "message": "IntelliPark API is running."}