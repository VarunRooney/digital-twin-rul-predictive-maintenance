# mqtt_publisher.py - simulates sensor data and publishes to MQTT broker (mosquitto)
import time, json, random
import paho.mqtt.client as mqtt
from datetime import datetime
client = mqtt.Client()
client.connect('localhost', 1883, 60)
client.loop_start()
try:
    temp=300.0; vib=0.2; curr=5.0; pressure=30.0
    while True:
        temp += 0.02 + random.gauss(0,0.2)
        vib += 0.001 + random.gauss(0,0.01)
        curr += 0.003 + random.gauss(0,0.05)
        pressure += 0.01 + random.gauss(0,0.05)
        payload = {'timestamp': datetime.utcnow().isoformat(), 'temp': temp, 'vib': vib, 'curr': curr, 'pressure': pressure}
        client.publish('factory/machine1/sensors', json.dumps(payload))
        time.sleep(1)
except KeyboardInterrupt:
    client.loop_stop()
