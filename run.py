from flask import Flask, render_template, Response, jsonify
import cv2
import os
from ultralytics import YOLO

app = Flask(__name__)

# Global variables for card counting and tracking
cardCount = 0

# Cards that add 1 to cardCount
plusOneCards = [
    '2C', '2D', '2H', '2S', '3C', '3D', '3H', '3S',
    '4C', '4D', '4H', '4S', '5C', '5D', '5H', '5S',
    '6C', '6D', '6H', '6S']

# Cards that subtract 1 from cardCount
minusOneCards = [
    '10C', '10D', '10H', '10S', 'JC', 'JD', 'JH', 'JS',
    'QC', 'QD', 'QH', 'QS', 'KC', 'KD', 'KH', 'KS',
    'AC', 'AD', 'AH', 'AS']

# Dictionary for tracking cards every frame, guarantees cards aren't misread
detection_tracker = {}

# Loads the YOLO model
model_path = "path to best.pt in training data"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")
model = YOLO(model_path)

#Video feed generator
def generate_frames():
    global cardCount, detection_tracker, plusOneCards, minusOneCards

    camera = cv2.VideoCapture(1)
    if not camera.isOpened():
        raise RuntimeError("Could not access camera.")

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        # Perform YOLO inference
        results = model.predict(frame, conf=0.5)
        for detection in results[0].boxes:
            x1, y1, x2, y2 = map(int, detection.xyxy[0])
            confidence = detection.conf[0]
            class_id = int(detection.cls[0])
            label_name = model.names[class_id].upper()

            # Track card detection
            if label_name not in detection_tracker:
                detection_tracker[label_name] = 0
            if confidence >= 0.6:
                detection_tracker[label_name] += 1
            if detection_tracker[label_name] >= 8:
                if label_name in plusOneCards:
                    cardCount += 1
                    plusOneCards.remove(label_name)
                elif label_name in minusOneCards:
                    cardCount -= 1
                    minusOneCards.remove(label_name)
                detection_tracker[label_name] = 0

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label_name} ({confidence:.2f})", 
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    camera.release()

# App routes
@app.route('/')
def index():
    return render_template('index.html', card_count=cardCount)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/card_count')
def card_count():
    global cardCount
    return jsonify({"cardCount": cardCount})

if __name__ == "__main__":
    app.run(debug=True)
