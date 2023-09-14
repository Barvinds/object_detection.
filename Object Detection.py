from flask import Flask, render_template, request, jsonify, Response
import cv2
import pyttsx3
from ultralytics import YOLO
import threading

app = Flask(__name__)

# Load the YOLOv5 model
model = YOLO("yolov8s.pt")

# Load class labels
class_labels = model.names

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Initialize camera and thread variables
cap = None
thread = None
stop_thread = False

def object_detection():
    global cap, stop_thread
    cap = cv2.VideoCapture(0)
    
    while not stop_thread:
        ret, frame = cap.read()
        
        if not ret:
            break

        # Perform object detection
        results = model.predict(frame)

        # Get detected labels from the first result
        first_result = results[0]
        detected_labels = first_result.boxes.cls.cpu().numpy().astype(int)

        if len(detected_labels) > 0:
            # Get class names for detected labels
            detected_classes = [class_labels[label] for label in detected_labels]
            speech_text = ", ".join(detected_classes) + " detected"

            # Generate speech output
            engine.say(speech_text)
            engine.runAndWait()

        # Encode the frame as JPEG
        _, img_encoded = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + img_encoded.tobytes() + b"\r\n")

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('obj_detect.html')

@app.route('/start_detection', methods=['GET'])
def start_detection():
    global thread
    if thread is None or not thread.is_alive():
        thread = threading.Thread(target=object_detection)
        thread.start()
    return jsonify({"status": "Detection started"})

@app.route('/stop_detection', methods=['GET'])
def stop_detection():
    global stop_thread
    stop_thread = True
    return jsonify({"status": "Detection stopped"})

@app.route('/video_feed')
def video_feed():
    return Response(object_detection(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == '__main__':
    app.run(debug=True)
