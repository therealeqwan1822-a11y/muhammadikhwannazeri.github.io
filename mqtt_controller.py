# mqtt_controller.py
import paho.mqtt.client as mqtt
import time
import threading

class HardwareDoorController:
    """
    Asynchronous MQTT controller for triggering virtual/physical 
    ESP32 door lock mechanisms on Wokwi.
    """
    def __init__(self, broker="broker.hivemq.com", port=1883, topic="attendance/door/command"):
        self.broker = broker
        self.port = port
        self.topic = topic
        self.client = mqtt.Client(client_id=f"PythonTASController-{int(time.time())}")
        self.is_connected = False

        # Configure Callbacks
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect

        # Start non-blocking network thread
        self.start()

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.is_connected = True
            print(f"[MQTT HARDWARE] Connected to HiveMQ Broker ({self.broker})")
        else:
            print(f"[MQTT HARDWARE ERROR] Connection failed with status code {rc}")

    def _on_disconnect(self, client, userdata, rc):
        self.is_connected = False

    def start(self):
        """Starts the MQTT loop in a background thread."""
        try:
            self.client.connect_async(self.broker, self.port, keepalive=60)
            self.client.loop_start()
        except Exception as e:
            print(f"[MQTT HARDWARE ERROR] Could not connect to broker: {e}")

    def unlock_door(self):
        """Sends unlock command ('1') to ESP32."""
        def _publish():
            try:
                result = self.client.publish(self.topic, "1", qos=1)
                result.wait_for_publish(timeout=2.0)
                print(f"[HARDWARE SIGNAL SENT] Door Unlock command ('1') -> {self.topic}")
            except Exception as e:
                print(f"[HARDWARE ERROR] Failed to send unlock signal: {e}")

        # Run publish in background thread to avoid freezing video processing
        threading.Thread(target=_publish, daemon=True).start()

    def lock_door(self):
        """Sends lock command ('0') to ESP32."""
        def _publish():
            try:
                result = self.client.publish(self.topic, "0", qos=1)
                result.wait_for_publish(timeout=2.0)
                print(f"[HARDWARE SIGNAL SENT] Door Lock command ('0') -> {self.topic}")
            except Exception as e:
                print(f"[HARDWARE ERROR] Failed to send lock signal: {e}")

        threading.Thread(target=_publish, daemon=True).start()

    def stop(self):
        """Disconnects and stops the MQTT client thread."""
        self.client.loop_stop()
        self.client.disconnect()