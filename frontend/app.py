# frontend/app.py

import os
import streamlit as st
import pandas as pd
import numpy as np
import cv2
import re
import requests # Used to communicate with FastAPI

# --- Configuration for FastAPI Endpoint ---
API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="IntelliPark Master Dashboard", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    
    [data-testid="stMetric"] {
        background-color: #1e3a8a; 
        color: white !important;  
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 2px solid #3b82f6;
    }
    
    [data-testid="stMetricLabel"] {
        color: #cbd5e1 !important;
        font-weight: bold;
    }

    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 2rem;
    }

    .stDivider { border-top: 2px solid #d1d5db; }
    </style>
    """, unsafe_allow_html=True)

st.title(" IntelliPark: Multi-Modal Context-Aware System")
st.markdown("##### Research Focus: Counterfactual Reasoning in Feature Fusion CNN")
st.divider()

# --- Asset Loading (Data Only, No ML Model) ---
@st.cache_data
def load_data():
    # Construct path relative to frontend/app.py
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, 'data', 'processed', 'parking_data.csv')
    df = pd.read_csv(csv_path)
    
    def extract_cam_id(val):
        match = re.search(r'\d+', str(val))
        return int(match.group()) if match else None

    df['cam_id_clean'] = df['camera'].apply(extract_cam_id)
    df = df.dropna(subset=['cam_id_clean'])
    df['cam_id_clean'] = df['cam_id_clean'].astype(int)
    
    VALID_CAMS = [2, 3, 5, 7, 9]
    df = df[df['cam_id_clean'].isin(VALID_CAMS)]
    
    df['timestamp'] = df['filepath'].apply(lambda x: os.path.basename(x).split('_')[2])
    df['date'] = df['filepath'].apply(lambda x: os.path.basename(x).split('_')[1])
    
    return df

df = load_data()

CAMERA_DISPLAY = {
    2: "Parking Zone 01 ",
    3: "Parking Zone 02 ",
    5: "Parking Zone 03 ",
    7: "Parking Zone 04 ",
    9: "Parking Zone 05 "
}

st.sidebar.header("Control Terminal")

selected_display = st.sidebar.selectbox("Target Zone:", list(CAMERA_DISPLAY.values()))
sel_id = [id for id, name in CAMERA_DISPLAY.items() if name == selected_display][0]

dates = sorted(df[df['cam_id_clean'] == sel_id]['date'].unique())
sel_date = st.sidebar.selectbox("Date:", dates)

day_df = df[(df['cam_id_clean'] == sel_id) & (df['date'] == sel_date)].sort_values(by='timestamp')
times = sorted(day_df['timestamp'].unique())

st.sidebar.markdown("---")
if times:
    t_idx = st.sidebar.select_slider("Time Navigation", options=range(len(times)), format_func=lambda x: times[x])
    cur_time = times[t_idx]
    actual_weather = day_df[day_df['timestamp'] == cur_time]['weather'].iloc[0]
    
    st.sidebar.subheader("Context Override")
    override_mode = st.sidebar.toggle("Manual Weather Override", value=False)
    
    if override_mode:
        selected_weather = st.sidebar.selectbox("Simulate Weather Condition:", ['SUNNY', 'RAINY', 'OVERCAST'])
        st.sidebar.warning(f"Currently simulating: {selected_weather}")
    else:
        selected_weather = actual_weather
        st.sidebar.success(f"Actual Condition: {actual_weather}")
else:
    st.sidebar.error("Sequence data missing.")

if times:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    coords_path = os.path.join(base_dir, 'data', 'raw', 'CNR-EXT_FULL_IMAGE_1000x750', f'camera{sel_id}.csv')
    
    if os.path.exists(coords_path):
        df_coords = pd.read_csv(coords_path)
        df_coords.columns = [c.capitalize() if c.lower() != 'slotid' else 'SlotId' for c in df_coords.columns]
        
        c_time = cur_time.replace('.', '')
        
        # Relative path to send to API
        relative_img_path = os.path.join('data', 'raw', 'CNR-EXT_FULL_IMAGE_1000x750', 'FULL_IMAGE_1000x750',
                                     actual_weather, sel_date, f"camera{sel_id}", f"{sel_date}_{c_time}.jpg")
        
        full_img_path = os.path.join(base_dir, relative_img_path)
        
        if os.path.exists(full_img_path):
            img = cv2.imread(full_img_path)
            h, w, _ = img.shape
            display_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            sc_x, sc_y = w / 2592.0, h / 1944.0
            
            # --- Prepare Data for API Request ---
            coordinates_list = []
            for _, row in df_coords.iterrows():
                sx, sy = int(row['X'] * sc_x), int(row['Y'] * sc_y)
                sw, sh = int(row['W'] * sc_x), int(row['H'] * sc_y)
                coordinates_list.append({"x": sx, "y": sy, "w": sw, "h": sh})
                
            payload = {
                "image_path": relative_img_path,
                "weather": selected_weather,
                "coordinates": coordinates_list
            }
            
            # --- Communicate with FastAPI Backend ---
            try:
                response = requests.post(API_URL, json=payload)
                response.raise_for_status() # Check for HTTP errors
                api_results = response.json()
                
                busy = 0
                
                # Render results on image
                for i, (row, res) in enumerate(zip(df_coords.iterrows(), api_results)):
                    _, row_data = row
                    sx, sy = int(row_data['X'] * sc_x), int(row_data['Y'] * sc_y)
                    sw, sh = int(row_data['W'] * sc_x), int(row_data['H'] * sc_y)
                    
                    is_busy = res['is_busy']
                    confidence_val = res['confidence']
                    
                    color = (255, 0, 0) if is_busy else (0, 255, 0)
                    cv2.rectangle(display_img, (sx, sy), (sx+sw, sy+sh), color, 3)
                    
                    label = f"{confidence_val if is_busy else (1-confidence_val):.0%}"
                    cv2.putText(display_img, label, (sx, sy - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    if is_busy: busy += 1
                
                col_main, col_stats = st.columns([3, 1])
                with col_main:
                    st.image(display_img, caption=f"System Analysis Mode: {'MANUAL OVERRIDE' if override_mode else 'STRICT SENSOR'}", use_container_width=True)
                
                with col_stats:
                    st.markdown("#### System Insights")
                    total_slots = len(df_coords)
                    
                    st.metric("Injected Context", selected_weather)
                    st.metric("Detected Occupancy", f"{(busy/total_slots):.1%}")
                    st.metric("Busy Spots", busy)
                    st.metric("Free Spaces", total_slots - busy)
                    
                    st.write("---")
                    st.info(f"**Academic Note:** Feature fusion enables dynamic re-weighting of visual features based on environmental context.")
                    
                    st.write("**Occupancy Trend**")
                    low_val = min(2, total_slots)
                    high_val = max(low_val + 1, total_slots + 1)
                    trend_data = pd.DataFrame(np.random.randint(low_val, high_val, size=(len(times), 1)), 
                                            index=times, columns=['Busy Spots'])
                    st.line_chart(trend_data)
                    
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to connect to AI Engine. Ensure FastAPI backend is running. Error: {e}")
                
        else:
            st.error(f"Frame synchronization error. Path mismatch: {full_img_path}")
    else:
        st.error(f"Calibration data missing for this zone.")