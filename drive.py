import socketio
import eventlet
import numpy as np
from flask import Flask
from keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2

sio = socketio.Server()

app = Flask(__name__)
speed_limit = 15

def img_preprocess(img):
    img = img[60:135, :, :]  # Crop the image to focus on road
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))  # Resize to model input shape
    img = img / 255.0  # Normalize
    return img

@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        speed = float(data['speed'])
        image = Image.open(BytesIO(base64.b64decode(data['image'])))
        image = np.asarray(image)
        image = img_preprocess(image)
        image = np.array([image])
        
        # Predict the steering angle
        steering_angle = float(model.predict(image))
        throttle = max(0.1, 1.0 - speed / speed_limit)
        
        print(f"Steering: {steering_angle}, Throttle: {throttle}, Speed: {speed}")
        
        send_control(steering_angle, throttle)
    else:
        print("No telemetry data received!")

@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    send_control(0, 0)

def send_control(steering_angle, throttle):
    sio.emit('steer', data={
        'steering_angle': str(steering_angle),
        'throttle': str(throttle)
    })

if __name__ == '__main__':
    model = load_model('D:\KKY_Brothers\Codes\Real Time Autonomous Car\model\model.h5')  # Fix path here
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('0.0.0.0', 4567)), app)
