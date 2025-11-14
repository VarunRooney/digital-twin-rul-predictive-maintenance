import paho.mqtt.client as mqtt
import pandas as pd
import json
from collections import deque
import threading
import time

# Configuration
MQTT_BROKER = "localhost"  # Since Mosquitto runs locally
MQTT_PORT = 1883
MQTT_TOPIC = "factory/machine1/sensors"
DATA_HISTORY_SIZE = 50  # Keep last 50 readings for the graph

# Global storage for sensor data (deque is thread-safe and efficient for pop/append)
sensor_data_history = deque(maxlen=DATA_HISTORY_SIZE)

# Lock for thread-safe access to the data deque
data_lock = threading.Lock()

# MQTT Callbacks
def on_connect(client, userdata, flags, rc):
    """Callback function for successful connection to MQTT broker."""
    if rc == 0:
        print("Connected to MQTT Broker!")
        client.subscribe(MQTT_TOPIC)
    else:
        print(f"Failed to connect, return code {rc}")

def on_message(client, userdata, msg):
    """Callback function executed every time a new message arrives."""
    try:
        # Decode and parse the JSON payload
        payload = json.loads(msg.payload.decode())
        
        # NOTE: The publisher uses 'temp', 'vib', 'pressure', 'curr'. 
        # We map these to the model's expected names: 'temperature', 'vibration', 'pressure'
        # We ignore 'curr' for now, as the RUL model wasn't trained on it.
        new_reading = {
            'timestamp': pd.Timestamp.utcnow(),
            'temperature': payload.get('temp'),
            'vibration': payload.get('vib'),
            'pressure': payload.get('pressure')
        }
        
        # Append data safely
        with data_lock:
            sensor_data_history.append(new_reading)
            
    except Exception as e:
        print(f"Error processing MQTT message: {e}")

# Core Streamlit data function
def get_mqtt_data_df() -> pd.DataFrame:
    """Returns the current sensor data history as a DataFrame."""
    with data_lock:
        if not sensor_data_history:
            # Return an empty dataframe with correct columns if no data received yet
            return pd.DataFrame(columns=['timestamp', 'temperature', 'vibration', 'pressure'])
        
        # Convert deque to DataFrame
        df = pd.DataFrame(list(sensor_data_history))
        return df

# Initialize MQTT Client and start connection loop in a separate thread
def start_mqtt_client():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_start()
    except ConnectionRefusedError:
        print("⚠️ Could not connect to MQTT broker. Ensure Docker is running!")
        # Fallback loop to prevent immediate script crash
        client.loop_stop()
        
# Start the MQTT connection thread immediately upon import
mqtt_thread = threading.Thread(target=start_mqtt_client, daemon=True)
mqtt_thread.start()