from ultralytics import YOLO
import cv2
import os

model_path = 'C:/Users/Dell/Documents/GitHub/IntelliPark_Project/models/best.pt'
image_path = 'C:/Users/Dell/Documents/GitHub/IntelliPark_Project/data/sri_lanka_plates' 
output_path = 'C:/Users/Dell/Documents/GitHub/IntelliPark_Project/results/test_detections'

os.makedirs(output_path, exist_ok=True)

model = YOLO(model_path)

results = model.predict(source=image_path, save=True, project='results', name='test_detections', conf=0.5)

print(f"--- Detection Test Complete ---")
print(f"Results saved in: results/test_detections/")