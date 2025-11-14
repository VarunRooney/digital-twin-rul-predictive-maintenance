import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time
from utils.mqtt_fetch import get_mqtt_data_df
from utils.predict_rul import predict_rul
from utils.forex_inr import get_inr_value

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Digital Twin | Real-Time Pro Dashboard", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. HELPER FUNCTIONS ---

# Function for Anomaly Detection
def check_anomaly(series: pd.Series, window: int = 5) -> bool:
    """Checks if the last point is outside 3 standard deviations of the moving average."""
    if len(series) < window: return False
    
    # Calculate moving average and standard deviation
    ma = series.rolling(window=window).mean().iloc[-2] # Second to last point for comparison
    std = series.rolling(window=window).std().iloc[-2]
    current_value = series.iloc[-1]
    
    # Simple 3-sigma check
    return (current_value > ma + 3 * std) or (current_value < ma - 3 * std)

# Function for RUL Health Status
def get_health_status(rul: float) -> tuple:
    """Returns status text and corresponding color."""
    if rul >= 70:
        return "HEALTHY", "green"
    elif 30 <= rul < 70:
        return "WARNING", "orange"
    else:
        return "CRITICAL FAILURE IMMINENT", "red"

# Function to calculate Expected Failure Date
def calculate_failure_date(rul_hours: float):
    if rul_hours <= 0:
        return "IMMEDIATE ACTION REQUIRED"
    failure_timestamp = pd.Timestamp.now() + pd.Timedelta(hours=rul_hours)
    return failure_timestamp.strftime("%Y-%m-%d %H:%M:%S IST")

# --- 3. UI SETUP ---
st.markdown("""
    <style>
    /* Customizing Streamlit's main content area */
    .stApp {
        background-color: #0e1117; 
        color: #f0f2f6; 
    }
    .stProgress > div > div { background-color: #2e8bff; }
    /* Dashboard Title */
    h1 { color: #5C95FF; } 
    /* Feature Badge Styling */
    .feature-badge {
        padding: 8px 15px;
        border-radius: 10px;
        font-weight: bold;
        color: white;
        text-align: center;
        margin-top: 10px;
        font-size: 16px;
    }
    .status-green { background-color: #28a745; }
    .status-orange { background-color: #ffc107; }
    .status-red { background-color: #dc3545; }
    .status-anomaly { background-color: #ff4b4b; } /* Bright Red for Anomaly */
    </style>
""", unsafe_allow_html=True)

st.title("⚙️ Digital Twin Pro Dashboard")
st.subheader("Real-Time RUL Prediction & Anomaly Detection")
st.markdown("---")

# --- 4. MAIN APP LOOP ---
placeholder = st.empty()
# FIX: Continuous loop for live data fetching
while True:
    # 1. Fetch live data (currently simulated, but ready for real-time API call)
    df = get_mqtt_data_df()
    
    # Ensure there is enough data for analysis
    if len(df) < 3: 
        st.warning("Waiting for sufficient sensor data points (need at least 3)...")
        time.sleep(1)
        continue

    # 2. Prediction & Cost
    rul_hours = predict_rul(df)
    inr_value = get_inr_value(rul_hours)
    status_text, status_color = get_health_status(rul_hours)
    failure_date = calculate_failure_date(rul_hours)
    
    # 3. Anomaly Checks on latest readings
    temp_anomaly = check_anomaly(df['temperature'])
    vib_anomaly = check_anomaly(df['vibration'])
    press_anomaly = check_anomaly(df['pressure'])
    
    # 4. Render Dashboard
    with placeholder.container():
        # --- ROW 1: RUL & COST METRICS ---
        col_rul, col_cost, col_failure = st.columns(3)
        
        # Predicted RUL
        col_rul.metric("Predicted Remaining Useful Life (RUL)", f"{rul_hours:.2f} hours")
        col_rul.markdown(f'<div class="feature-badge status-{status_color}">{status_text}</div>', unsafe_allow_html=True)
        
        # Estimated Cost
        col_cost.metric("Estimated Maintenance Cost", f"₹{inr_value:,.2f}")
        col_cost.markdown('<div class="feature-badge status-green">Cost Projection</div>', unsafe_allow_html=True)
        
        # Predictive Timeline
        col_failure.metric("Expected Failure Date", failure_date)
        col_failure.markdown('<div class="feature-badge status-orange">Predictive Timeline</div>', unsafe_allow_html=True)
        
        st.markdown("---")

        # --- ROW 2: LIVE SENSOR READINGS & ANOMALY ---
        st.subheader("Live Sensor Readings & Anomaly Check")
        col_temp, col_vib, col_press = st.columns(3)

        def render_sensor_metric(col, name, value, is_anomaly):
            anomaly_icon = "🚨" if is_anomaly else "🟢"
            anomaly_class = "status-anomaly" if is_anomaly else "status-green"
            col.metric(f"{name}", f"{value:.2f}")
            col.markdown(f'<div class="feature-badge {anomaly_class}">{anomaly_icon} {"Anomaly Detected" if is_anomaly else "Normal"}</div>', unsafe_allow_html=True)

        render_sensor_metric(col_temp, "Temperature (°C)", df['temperature'].iloc[-1], temp_anomaly)
        render_sensor_metric(col_vib, "Vibration (mm/s)", df['vibration'].iloc[-1], vib_anomaly)
        render_sensor_metric(col_press, "Pressure (bar)", df['pressure'].iloc[-1], press_anomaly)

        st.markdown("---")

        # --- ROW 3: TIME-SERIES GRAPH ---
        st.subheader("Sensor Trend Chart")
        
        fig = go.Figure()
        # NOTE: Using simple index for X-axis since live timestamp is not confirmed
        fig.add_trace(go.Scatter(y=df['temperature'], name="Temperature", mode='lines', line=dict(color='#2e8bff')))
        fig.add_trace(go.Scatter(y=df['vibration'], name="Vibration", mode='lines', line=dict(color='#ff7f0e')))
        fig.add_trace(go.Scatter(y=df['pressure'], name="Pressure", mode='lines', line=dict(color='#2ca02c')))
        
        fig.update_layout(
            template='plotly_dark', 
            height=450,
            xaxis_title="Time Steps (or Timestamp from Live Feed)",
            yaxis_title="Sensor Value"
        )
        st.plotly_chart(fig, use_container_width=True)

    # 5. Update frequency
    time.sleep(2)