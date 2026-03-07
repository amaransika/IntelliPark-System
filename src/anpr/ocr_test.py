import easyocr
import pandas as pd
import re

def run_ocr_security_check(image_path, flagged_csv_path):
    reader = easyocr.Reader(['en'], gpu=False) 

    print("--- Character Recognition & Security Check ---")
    
    results = reader.readtext(image_path)
    
    detected_text_list = []
    total_confidence = 0
    
    for (bbox, text, prob) in results:
        clean_chunk = re.sub(r'[^A-Za-z0-9]', '', text)
        detected_text_list.append(clean_chunk)
        total_confidence += prob
        print(f"Detected Chunk: {text} (Confidence: {prob:.2f})")

    full_plate = "".join(detected_text_list).upper()
    avg_confidence = total_confidence / len(results) if results else 0
    
    print(f"\nFinal Combined Plate: {full_plate}")
    print(f"Average Confidence: {avg_confidence:.2f}")

    try:
        flagged_df = pd.read_csv(flagged_csv_path)
        flagged_list = flagged_df['plate_number'].astype(str).str.upper().str.replace(" ", "").tolist()
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    if full_plate in flagged_list:
        print(f"Security Status: ❌ FLAGGENED - THREAT DETECTED")
        print("Action: Entry Blocked. Alert sent to Dashboard.")
    else:
        print(f"Security Status: ✅ AUTHORIZED")
        print("Action: Entry Permitted. Querying GNN for spot assignment...")

    print("-" * 45)

if __name__ == "__main__":
    IMAGE_PATH = "C:/Users/Dell/Documents/GitHub/IntelliPark_Project/data/test_plate.jpg" 
    CSV_PATH = "C:/Users/Dell/Documents/GitHub/IntelliPark_Project/data/flagged_vehicles.csv"
    run_ocr_security_check(IMAGE_PATH, CSV_PATH)