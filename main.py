import threading
from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import time
import dlib
import cv2
import pyttsx3
from flask import Flask, render_template, Response, request


def alert_driver():
    v = pyttsx3.init()
    voices = v.getProperty('voices')
    v.setProperty("rate", 178)
    v.setProperty('voice', voices[1].id)
    v.say("Alert: You seem to be drowsy. Please take a break.")
    v.runAndWait()

def alert_driver_thread():
    alert_thread = threading.Thread(target=alert_driver)
    alert_thread.start()


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)

    return ear


def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)


def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance



YAWN_THRESH = 14
alarm_status = False
alarm_status2 = False
saying = False
global COUNTER
global YAWN_COUNTER
global EYE_AR_THRESH
global EYE_AR_CONSEC_FRAMES

EYE_AR_THRESH = 0.26
EYE_AR_CONSEC_FRAMES = 45
SWITCH = 0
COUNTER = 0
YAWN_COUNTER = 0


detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')




def camera_function():
    global COUNTER, YAWN_COUNTER, EYE_AR_THRESH, EYE_AR_CONSEC_FRAMES
    while True:
        
        success,frame = camera.read()
        rects = detector.detectMultiScale(frame,1.1,7)
        
        for (x, y, w, h) in rects:
            rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            
            shape = predictor(frame, rect)
            shape = face_utils.shape_to_np(shape)

            eye = final_ear(shape)
            
            ear = eye[0]
            leftEye = eye[1]
            rightEye = eye[2]

            distance = lip_distance(shape)
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            lip = shape[48:60]
            cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

            if ear < EYE_AR_THRESH:
                COUNTER += 1

                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    alert_driver_thread()
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    COUNTER = 0


            if (distance > YAWN_THRESH):
                YAWN_COUNTER+=1
                if YAWN_COUNTER%3 == 0 and YAWN_COUNTER!=0 :
                    alert_driver_thread
                    cv2.putText(frame, "Yawn Alert", (10, 130),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    YAWN_COUNTER = 0
                    

            cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "YAWN: {:.2f}".format(distance), (300, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "LEFTEYE: {:.2f}".format(eye_aspect_ratio(leftEye)), (0, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "RIGHTEYE: {:.2f}".format(eye_aspect_ratio(rightEye)), (0, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
                pass


app = Flask(__name__, template_folder='./templates')



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(camera_function(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/requests',methods=['POST','GET'])
def tasks():
    global SWITCH,camera
    if request.method == 'POST':  
        if request.form.get('stop') == 'Stop/Start':
            if(SWITCH==1):
                SWITCH=0
                camera.release()
                cv2.destroyAllWindows()
                print("camera stopping")
                
            else:
                print("starting camera")
                camera = cv2.VideoCapture(0)
                SWITCH=1

    elif request.method=='GET':
        return render_template('index.html')
    return render_template('index.html')



app.run(debug=True)