# ⚙️ Digital Twin: Predictive Maintenance for Rotating Machinery

This project implements a **Digital Twin** framework for predictive maintenance, using real-time sensor simulation, MQTT for data streaming, and machine learning (Random Forest/LSTM) to predict the **Remaining Useful Life (RUL)** of a machine. The data is visualized via a Streamlit dashboard.

---

## 🚀 Key Technologies

* **Data Stream:** **MQTT** (Mosquitto Docker Broker)
* **Data Simulation:** **Python** (`mqtt_publisher.py`)
* **Predictive Models:** **Scikit-learn** (Random Forest for RUL & Isolation Forest for Anomaly Detection) or **PyTorch** (LSTM)
* **Visualization:** **Streamlit** Dashboard
* **Containerization:** **Docker** and **Docker Compose** (Used for Mosquitto, Prometheus, and Grafana)

---

## 🛠️ Setup and Prerequisites

Before launching the project, ensure you have the following installed:

1.  **Conda/Miniconda:** For managing the Python environment.
2.  **Docker Desktop:** For running the MQTT broker (Mosquitto).
3.  **Git:** (Optional, but recommended for cloning the repository).

## 🧩 Project Structure
digital_twin_rul_pro/
│
├── app.py                       # Streamlit dashboard
├── models/
│ └── lstm_rul_model.pt          # Trained AI model
├── data/
│ └── engine_sensor_stream.csv   # Example sensor dataset
├── utils/
│ ├── data_loader.py
│ ├── preprocess.py
│ ├── simulate_feed.py
│ ├── predict_rul.py
│ └── forex_inr.py
│ └── mqtt_fetch.py
├── infra/
│ └── mosquitto/
      └── mosquitto.conf  
├── Dockerfile
├── lstm_rul_model.pt
├── mqtt_publisher.py
├── mqtt_subscriber.py
├── requirements.txt
└── README.md


---

## 🧠 Tech Stack
| Component | Technology |
|------------|-------------|
| Programming Language | Python 3.10 |
| AI Framework | PyTorch 2.0 |
| Dashboard | Streamlit |
| Visualization | Plotly |
| Data Handling | Pandas, NumPy |
| API | Open Exchange Rates (USD→INR) |

---

## 🚀 Setup Instructions (VS Code + Conda)

💻 Project Setup and Launch CommandsPhase

 1: Initial Setup and Dependencies (New Laptop)These commands set up the project structure and ensure all Python dependencies are installed correctly inside your dedicated environment.
 
 Steps:
 1. Change Directory     -  cd C:\Users\YourUser\digital_twin_rul_pro  #Navigate into your project folder.
 2. Create Environment   - conda create -n twin python=3.10   #Create a clean Conda environment named twin.
 3. Activate Environment - conda activate twin    #Switch to the dedicated environment.
 4. Install Dependencies - pip install -r requirements.txt   #Install all required Python libraries (e.g., Streamlit, pandas, numpy).
 5. Install Paho-MQTT    - pip install paho-mqtt   #Ensure the MQTT client library is explicitly installed.
 6. Train RUL Model      - python train_model.py    #This trains the Random Forest model and saves model.pkl, scaler.pkl, and anomaly.pkl to the app/ folder.
 
Phase 2: 
Launching the Real-Time System (Three Separate Terminals)You must use three separate terminal windows to run these commands simultaneously.

1️⃣ Terminal 1: MQTT Broker (Docker)
This command starts the Mosquitto broker, which acts as the real-time data hub.

1.  cd C:\Users\YourUser\digital_twin_rul_pro

2.  docker compose down

3.  docker compose up -d mosquitto

4.  docker ps  # VERIFY the Mosquitto container shows STATUS "Up"

2️⃣ Terminal 2: Data Publisher (Python)
This terminal starts sending live, simulated sensor data to the running broker.Bashconda activate twin

1.  cd C:\Users\YourUser\digital_twin_rul_pro
2.  python mqtt_publisher.py

# Leave this terminal running.

3️⃣ Terminal 3: Dashboard (Streamlit)
This terminal starts the dashboard, which subscribes to the broker and displays the RUL prediction and charts.Bashconda activate twin

1.  cd C:\Users\YourUser\digital_twin_rul_pro

2.  streamlit run app.py

The Streamlit app will open in your browser (usually http://localhost:8501). The RUL and charts will update after about 20 seconds.


👨‍💻 Author

Varun Kumar R
AIML Department, Vemana Institute of Technology
