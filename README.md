# 🚗 IntelliPark: AI-Powered Smart Parking & ANPR Security System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-red)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-yellow)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)

Welcome to **IntelliPark**, a fully automated, context-aware smart parking and security gateway system. This project eliminates the need for expensive hardware sensors by utilizing advanced Computer Vision and Artificial Intelligence to monitor parking availability and automate vehicle entry via License Plate Recognition.

## ✨ Key Features

* **⛅ Condition-Gated Parking Detection (CNN):** A custom Multi-Modal CNN that takes environmental weather (Sunny, Rainy, Overcast) into context to accurately predict empty slots without being confused by shadows or wet reflections.
* **🔎 Advanced ANPR Gateway:** Uses a custom-trained **YOLOv8** model combined with **EasyOCR** to detect and read license plates.
* **🇱🇰 Sri Lankan Context Optimization:** Features a custom Regex algorithm to automatically filter out unnecessary province letters (e.g., WP, CP), improving read speed and accuracy.
* **⚡ Zero-Latency Asynchronous Logging:** Utilizes Python's `asyncio` to log vehicle data to a **Supabase Cloud Database** in the background, ensuring the live CCTV video feed never freezes.
* **📊 Interactive Web Dashboard:** A user-friendly **Streamlit** frontend for facility managers and security guards to monitor live video feeds and receive real-time alerts.

---

## 🛠️ Technology Stack

* **Backend & API:** FastAPI, Uvicorn, Python `asyncio`
* **Frontend UI:** Streamlit
* **AI & Machine Learning:** TensorFlow/Keras (CNN), Ultralytics YOLOv8, EasyOCR, OpenCV
* **Database:** Supabase (PostgreSQL), Local CSV logging

---

## ⚙️ Environment Setup & Installation

Follow these step-by-step instructions to set up the IntelliPark system on your local machine.

### 1. Prerequisites
Make sure you have the following installed:
* [Python 3.9 or higher](https://www.python.org/downloads/)
* [Git](https://git-scm.com/)
* A Supabase account (for cloud database logging)

### 2. Clone the Repository
Open your terminal or command prompt and run:
```bash
git clone [https://github.com/yourusername/IntelliPark.git](https://github.com/yourusername/IntelliPark.git)
cd IntelliPark
```
*(Note: Replace the URL with your actual repository link)*

### 3. Create a Virtual Environment
It is highly recommended to use a virtual environment to keep dependencies organized and avoid conflicts.
```bash
# Create the virtual environment named 'venv'
python -m venv venv

# Activate the virtual environment
# For Windows:
venv\Scripts\activate

# For macOS/Linux:
source venv/bin/activate
```

### 4. Install Dependencies
Once the virtual environment is activated, install all the required Python libraries. 
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
*(Note: If you have a dedicated GPU, make sure to install the CUDA-compatible versions of PyTorch and TensorFlow for faster AI processing).*

### 5. Configure Environment Variables (.env)
The system requires a connection to Supabase to log ANPR data.
1. Create a new file named `.env` in the root directory of the project.
2. Add your Supabase credentials to the `.env` file like this:
```env
SUPABASE_URL=your_supabase_project_url_here
SUPABASE_KEY=your_supabase_api_key_here
```

---

## 🚀 How to Run the Application

The IntelliPark system runs on a Decoupled 2-Tier Architecture. You need to start the backend AI engine and the frontend dashboard separately.

### Step 1: Start the FastAPI Backend (AI Engine)
Keep your virtual environment activated and run the following command to start the backend server:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
*The backend will start running at `http://localhost:8000`. This controls the AI models and the MJPEG video stream.*

### Step 2: Start the Streamlit Dashboard (Frontend)
Open a **new terminal window**, activate the virtual environment again, and run:
```bash
# Activate env again in the new terminal
venv\Scripts\activate   # (Windows)
# OR source venv/bin/activate (macOS/Linux)

# Start Streamlit
streamlit run app.py
```
*The frontend dashboard will automatically open in your default web browser at `http://localhost:8501`.*

---

## 📂 Project Structure

```text
IntelliPark/
│
├── models/                  # Saved AI models (.keras, .pt files)
├── data/                    # Sample videos and CSV logs
├── backend/                 # FastAPI routes and AI processing logic
│   ├── main.py              # Main API entry point
│   ├── anpr_engine.py       # YOLOv8 + EasyOCR logic
│   └── cnn_engine.py        # Parking slot detection logic
│
├── frontend/                # Streamlit UI components
│   └── app.py               # Main Streamlit dashboard script
│
├── requirements.txt         # List of Python dependencies
├── .env                     # Environment variables (Do not commit this file!)
└── README.md                # Project documentation
```

---

## 👨‍💻 Author
Developed as a comprehensive academic and practical solution to modernize facility management.
