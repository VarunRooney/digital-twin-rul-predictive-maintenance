# mqtt_subscriber.py - subscribes to sensor topic and prints messages
import paho.mqtt.client as mqtt, json
def on_connect(client, userdata, flags, rc):
    client.subscribe('factory/+/sensors')
def on_message(client, userdata, msg):
    print(msg.topic, msg.payload.decode())
c = mqtt.Client()
c.on_connect = on_connect
c.on_message = on_message
c.connect('localhost', 1883, 60)
c.loop_forever()
