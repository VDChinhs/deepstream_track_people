import Kafka.config
from kafka import KafkaProducer
import json
import cv2
import base64

topic_name = 'cam'

PRODUCER_SERVICE = None

def get_producer():
    global PRODUCER_SERVICE
    if PRODUCER_SERVICE is None:
        PRODUCER_SERVICE = KafkaProducer(
            bootstrap_servers = [Kafka.config.kafka_ip]
        )
    return PRODUCER_SERVICE

def encode_image(image):
    _, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return buffer.tobytes()

def send_image(image, gate, typemove):
    try:
        image_bytes = encode_image(image)
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        data = {
            "gate": gate,
            "image_data": image_base64,
            "type": typemove
        }
        data_bytes = json.dumps(data).encode('utf-8')
        # print(data_bytes)
        get_producer().send(topic_name, value=data_bytes)
        get_producer().flush()
    except Exception as e:
        print(f"Error send data: {e}")

# image_path = '/home/jetsonvy/DucChinh/frame1100.jpg'
# img = cv2.imread(image_path)
# image_bytes = encode_image(img)
# get_producer().send(topic_name, image_bytes)
# print(f"Image sent: {image_path}")
# get_producer().flush()
