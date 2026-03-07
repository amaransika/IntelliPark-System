# frontend/app.py

import os
import streamlit as st
import pandas as pd
import numpy as np
import cv2
import re
import requests
from streamlit_autorefresh import st_autorefresh

# --- 1. Page Configuration & Navigation Logic ---
st.set_page_config(page_title="IntelliPark Master Dashboard", layout="wide", page_icon="🚗")

if 'page' not in st.session_state:
    st.session_state.page = 'dashboard'

# Auto refresh only on the ANPR page (every 1 second)
if st.session_state.page == 'anpr':
    st_autorefresh(interval=1000, limit=None, key="anpr_refresh")

BACKEND_BASE = os.getenv("BACKEND_API_URL", "http://127.0.0.1:8000").replace("/predict", "")

# --- 2. Custom CSS Styles ---
st.markdown("""
    <style>
    .main { background-color: #0f172a; color: #f1f5f9; } /* Dark slate background like React */
    
    /* Parking Dashboard Styling */
    [data-testid="stMetric"] {
        background-color: #1e293b; 
        color: white !important;  
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        border: 1px solid #334155;
    }
    [data-testid="stMetricLabel"] { color: #94a3b8 !important; font-weight: bold; text-transform: uppercase; letter-spacing: 1px; }
    [data-testid="stMetricValue"] { color: #ffffff !important; font-size: 2rem; }

    /* Custom Alert Banner */
    .red-banner {
        background-color: #dc2626; color: white; text-align: center;
        padding: 15px; font-weight: 900; font-size: 24px; letter-spacing: 3px;
        border-radius: 8px; margin-bottom: 20px; box-shadow: 0 0 20px rgba(220, 38, 38, 0.6);
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.8; }
        100% { opacity: 1; }
    }
    
    /* Log Entries */
    .log-entry-safe { background-color: #022c22; color: #34d399; padding: 10px; border-radius: 5px; margin-bottom: 5px; border: 1px solid #064e3b; font-family: monospace; }
    .log-entry-alert { background-color: #450a0a; color: #fca5a5; padding: 10px; border-radius: 5px; margin-bottom: 5px; border: 1px solid #7f1d1d; font-family: monospace; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. Sidebar Navigation & Alerts ---
st.sidebar.title("🎮 Navigation")
st.sidebar.markdown("---")
st.sidebar.error("📢 **Security Alert System**")
if st.sidebar.button("🚨 VIEW LIVE ANPR ALERTS", use_container_width=True):
    st.session_state.page = 'anpr'
if st.sidebar.button("🏠 BACK TO DASHBOARD", use_container_width=True):
    st.session_state.page = 'dashboard'
st.sidebar.markdown("---")


# =========================================================
# 4. PARKING DASHBOARD LOGIC (Original Intact)
# =========================================================
if st.session_state.page == 'dashboard':
    st.title("🚗 IntelliPark: Multi-Modal Context-Aware System")
    st.markdown("##### Research Focus: Counterfactual Reasoning in Feature Fusion CNN")
    st.divider()

    @st.cache_data
    def load_data():
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(base_dir, 'data', 'processed', 'parking_data.csv')
        df = pd.read_csv(csv_path)
        
        df['cam_id_temp'] = df['camera'].astype(str).str.extract(r'(\d+)')
        df = df.dropna(subset=['cam_id_temp'])
        df['cam_id_clean'] = df['cam_id_temp'].astype(int)
        
        VALID_CAMS = [2, 3, 5, 7, 9]
        df = df[df['cam_id_clean'].isin(VALID_CAMS)]
        
        df['timestamp'] = df['filepath'].apply(lambda x: os.path.basename(x).split('_')[2])
        df['date'] = df['filepath'].apply(lambda x: os.path.basename(x).split('_')[1])
        return df

    df = load_data()

    CAMERA_DISPLAY = {2: "Parking Zone 01", 3: "Parking Zone 02", 5: "Parking Zone 03", 7: "Parking Zone 04", 9: "Parking Zone 05"}
    st.sidebar.header("Control Terminal")
    selected_display = st.sidebar.selectbox("Target Zone:", list(CAMERA_DISPLAY.values()))
    sel_id = [id for id, name in CAMERA_DISPLAY.items() if name == selected_display][0]

    dates = sorted(df[df['cam_id_clean'] == sel_id]['date'].unique())
    sel_date = st.sidebar.selectbox("Date:", dates)
    day_df = df[(df['cam_id_clean'] == sel_id) & (df['date'] == sel_date)].sort_values(by='timestamp')
    times = sorted(day_df['timestamp'].unique())

    if times:
        t_idx = st.sidebar.select_slider("Time Navigation", options=range(len(times)), format_func=lambda x: times[x])
        cur_time = times[t_idx]
        actual_weather = day_df[day_df['timestamp'] == cur_time]['weather'].iloc[0]
        
        st.sidebar.subheader("Context Override")
        override_mode = st.sidebar.toggle("Manual Weather Override", value=False)
        selected_weather = st.sidebar.selectbox("Simulate Weather Condition:", ['SUNNY', 'RAINY', 'OVERCAST']) if override_mode else actual_weather
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        coords_path = os.path.join(base_dir, 'data', 'raw', 'CNR-EXT_FULL_IMAGE_1000x750', f'camera{sel_id}.csv')
        
        if os.path.exists(coords_path):
            df_coords = pd.read_csv(coords_path)
            df_coords.columns = [c.capitalize() if c.lower() != 'slotid' else 'SlotId' for c in df_coords.columns]
            
            c_time = cur_time.replace('.', '')
            relative_img_path = os.path.join('data', 'raw', 'CNR-EXT_FULL_IMAGE_1000x750', 'FULL_IMAGE_1000x750', actual_weather, sel_date, f"camera{sel_id}", f"{sel_date}_{c_time}.jpg")
            full_img_path = os.path.join(base_dir, relative_img_path)
            
            if os.path.exists(full_img_path):
                img = cv2.imread(full_img_path)
                h, w, _ = img.shape
                display_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                sc_x, sc_y = w / 2592.0, h / 1944.0
                
                coordinates_list = [{"x": int(r['X'] * sc_x), "y": int(r['Y'] * sc_y), "w": int(r['W'] * sc_x), "h": int(r['H'] * sc_y)} for _, r in df_coords.iterrows()]
                payload = {"image_path": relative_img_path, "weather": selected_weather, "coordinates": coordinates_list}
                
                try:
                    response = requests.post(f"{BACKEND_BASE}/predict", json=payload, timeout=120)
                    response.raise_for_status() 
                    api_results = response.json()
                    
                    busy = 0
                    for i, (row, res) in enumerate(zip(df_coords.iterrows(), api_results)):
                        _, row_data = row
                        sx, sy = int(row_data['X'] * sc_x), int(row_data['Y'] * sc_y)
                        sw, sh = int(row_data['W'] * sc_x), int(row_data['H'] * sc_y)
                        
                        is_busy, confidence_val = res['is_busy'], res['confidence']
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
                except requests.exceptions.RequestException as e:
                    st.error(f"Failed to connect to AI Engine. Error: {e}")
            else: st.error(f"Path mismatch: {full_img_path}")
        else: st.error(f"Calibration data missing for this zone.")
    else: st.error("Sequence data missing.")


# =========================================================
# 5. ANPR PAGE LOGIC (React-Style UI in Streamlit)
# =========================================================
elif st.session_state.page == 'anpr':
    
    # Fetch data from API
    try:
        api_data = requests.get(f"{BACKEND_BASE}/api/anpr/status", timeout=2).json()
        anpr_feed = api_data.get("feed", {})
        history = api_data.get("history", [])
    except:
        anpr_feed = {"best_plate": "--", "confidence": 0.0, "is_flagged": False, "alert_message": "Connecting..."}
        history = []

    # 1. Massive Red Alert Banner (Shows only if flagged)
    if anpr_feed.get("is_flagged", False):
        st.markdown('<div class="red-banner">🚨 SECURITY ALERT: BLACKLISTED VEHICLE DETECTED 🚨</div>', unsafe_allow_html=True)
    
    st.markdown('<h1 style="color: #60a5fa;">🛡️ ANPR Gateway Sentinel</h1>', unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8;'>Live Optical Character Recognition & DB Matching</p>", unsafe_allow_html=True)
    st.divider()

    col_vid, col_data = st.columns([2.5, 1])
    
    # 2. Live Video Feed
    with col_vid:
        st.markdown("<h4 style='color: #e2e8f0;'>🎥 Live Entrance Feed</h4>", unsafe_allow_html=True)
        # We display the raw streaming endpoint URL as an image
        st.image(f"{BACKEND_BASE}/api/anpr/video_feed", use_container_width=True)

    # 3. Info and History Panel
    with col_data:
        # Status Box
        plate_color = "#ef4444" if anpr_feed.get("is_flagged", False) else "#34d399" if anpr_feed.get("best_plate") != "--" else "#475569"
        
        st.markdown(f"""
        <div style="background-color: #0f172a; padding: 20px; border-radius: 10px; border: 2px solid {plate_color}; text-align: center;">
            <p style="color: #94a3b8; font-size: 12px; letter-spacing: 2px; text-transform: uppercase;">Target Plate Locked</p>
            <h2 style="color: {plate_color}; font-family: monospace; font-size: 40px; margin: 10px 0;">{anpr_feed.get('best_plate', '--')}</h2>
            <p style="color: #cbd5e1; font-size: 14px;">OCR Confidence: <b>{(anpr_feed.get('confidence', 0)*100):.1f}%</b></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Recent Scans History
        st.markdown("<h4 style='color: #e2e8f0; border-bottom: 1px solid #334155; padding-bottom: 10px;'>📋 Recent Scans</h4>", unsafe_allow_html=True)
        
        if len(history) == 0:
            st.markdown("<p style='color: #64748b; font-family: monospace;'>Waiting for vehicles...</p>", unsafe_allow_html=True)
        else:
            for log in history:
                if log['flagged']:
                    st.markdown(f"<div class='log-entry-alert'>🚨 {log['plate']} <span style='float:right; color:#9ca3af; font-size:12px;'>{log['time']}</span></div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='log-entry-safe'>✅ {log['plate']} <span style='float:right; color:#64748b; font-size:12px;'>{log['time']}</span></div>", unsafe_allow_html=True)