import cv2
import math
import numpy as np
import hand_tracking

from flask import Flask, render_template, Response


detector = hand_tracking.HandDetector(min_detection_confidence=0.7)
cap = cv2.VideoCapture(0)

def generate_frames():   

    vol_bar = 0
    while True:
        # Capture frame-by-frame.
        ret, frame = cap.read()
        frame = detector.find_hands(frame, draw=False)
        landmark_list = detector.find_positions(frame, draw=False)

        if landmark_list:
            x1, y1 = landmark_list[4][1], landmark_list[4][2]
            x2, y2 = landmark_list[8][1], landmark_list[8][2]

            #cv2.circle(frame, (x1, y1), 5, (255, 0, 0), cv2.FILLED)
            #cv2.circle(frame, (x2, y2), 5, (255, 0, 0), cv2.FILLED)
            #cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

            length = math.hypot(x2-x1, y2-y1)

            vol_bar = np.interp(length, [50, 300], [400, 170])

        else:
            vol_bar = 400

        cv2.rectangle(frame, (50, 170), (85, 400), (255, 0, 0), 3)
        cv2.rectangle(frame, (50, int(vol_bar)), (85, 400), (255, 0, 0), 
                    cv2.FILLED)


        flag, buffer = cv2.imencode('.jpg', frame)

        # Ensure the frame was successfully encoded.
        if not flag:
            continue

        frame = buffer.tobytes()

        # Concat frame one by one and show result.
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')




app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/video_feed', methods=['POST'])
def video_feed():
    return Response(generate_frames(), 
        mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)


