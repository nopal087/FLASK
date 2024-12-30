from flask import Flask, render_template, request, jsonify
import json
import cv2
import numpy as np
from keras.preprocessing import image
import warnings
from keras.models import load_model
import base64
import webbrowser
import os

warnings.filterwarnings("ignore")

app = Flask(__name__)

# Load the model
model_path = os.path.join(os.path.dirname(__file__), "mobilenet_model.h5")
model = load_model(model_path)

# Initialize face detector using Haar Cascade
face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

emotion_counts = {emotion: 0 for emotion in emotions}

# Default YouTube links
youtube_links = {
    'angry': "https://www.youtube.com/watch?v=i90EvEZ3axw&list=PL97kH0xIAu5n8JTmEVhxCSp8EhIrAXO8R&pp=gAQB",
    'fear': "https://www.youtube.com/watch?v=d5gf9dXbPi0&list=PL97kH0xIAu5k2e4Uynb1mS6E0gdSyy8Gx&pp=gAQB",
    'happy': "https://www.youtube.com/watch?v=PEM0Vs8jf1w&list=PL97kH0xIAu5lRrHRIr-OlINoAGC3yNjs6&pp=gAQB",
    'sad': "https://www.youtube.com/watch?v=_m6l5nKEGIA&list=PL97kH0xIAu5kcX185p6-u8nr6boWXLZNH&pp=gAQB",
    'neutral': "https://www.youtube.com/watch?v=dBFp0Ext0y8&list=PL97kH0xIAu5llG7M9KG6AANgjHE5vpHv2&pp=gAQB"
}


def load_links():
    global youtube_links
    try:
        with open('links.json', 'r') as file:
            youtube_links = json.load(file)
    except FileNotFoundError:
        pass


@app.before_request
def reload_links():
    load_links()


@app.route('/')
def index():
    return render_template('check_page.html')


@app.route('/get_link/<emotion>', methods=['GET'])
def get_link(emotion):
    if emotion in youtube_links:
        return jsonify({'success': True, 'link': youtube_links[emotion]})
    return jsonify({'success': False, 'message': 'Emotion not found'})


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        img_data = base64.b64decode(data['image'].split(',')[1])
        np_img = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces_detected = face_haar_cascade.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=5)

        if len(faces_detected) == 0:
            return jsonify({'emotion': 'No face detected'})

        x, y, w, h = faces_detected[0]
        roi_gray = gray_img[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (224, 224))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)
        max_index = np.argmax(predictions[0])
        predicted_emotion = emotions[max_index]

        emotion_counts[predicted_emotion] += 1

        if sum(emotion_counts.values()) == 15:
            highest_emotion = max(emotion_counts, key=emotion_counts.get)
            if highest_emotion in youtube_links:
                webbrowser.open(youtube_links[highest_emotion])
            return jsonify({'emotion': predicted_emotion, 'done': True})

        return jsonify({'emotion': predicted_emotion, 'done': False})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/reset', methods=['POST'])
def reset():
    global emotion_counts
    emotion_counts = {emotion: 0 for emotion in emotions}
    return jsonify({'reset': True})


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
