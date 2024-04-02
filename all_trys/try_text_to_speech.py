import cv2
import numpy as np
import face_recognition
import os
from djitellopy import Tello
import time
import pyttsx3

# Initialize Tello drone
me = Tello()
me.connect()
print(me.get_battery())
me.streamon()
time.sleep(2.2)

# Constants and variables for face tracking
w, h = 360, 240
fbRange = [6200, 6800]
pid = [0.4, 0.4, 0]
pError = 0

# Load images for face recognition
path = 'ImagesAttendance'
images = []
classNames = []
myList = os.listdir(path)
for cls in myList:
    curImg = cv2.imread(f'{path}/{cls}')
    images.append(curImg)
    classNames.append(os.path.splitext(cls)[0])

encodeListKnown = []

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)
print('Encoding Complete')

# Function to find a recognized face and return its name
def recognizeFace(img, encodeListKnown, classNames):
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            return name, faceLoc
    return None, None

# Function to track a face
def trackFace(info, w, pid, pError):
    area = info[1]
    x, y = info[0]
    fb = 0
    error = x - w // 2
    speed = pid[0] * error + pid[1] * (error - pError)
    speed = int(np.clip(speed, -100, 100))
    if area > fbRange[0] and area < fbRange[1]:
        fb = 0
    elif area > fbRange[1]:
        fb = -20
    elif area < fbRange[0] and area != 0:
        fb = 20

    if x == 0:
        speed = 0
        error = 0

    me.send_rc_control(0, fb, 0, speed)
    return error

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Main loop for integrated functionality
while True:
    img = me.get_frame_read().frame
    img = cv2.resize(img, (w, h))

    # Recognize face
    name, faceLoc = recognizeFace(img, encodeListKnown, classNames)

    if name is not None:
        # Face recognized, track the face
        x1, y1, x2, y2 = faceLoc
        info = [[(x1 + x2) // 2, (y1 + y2) // 2], (x2 - x1) * (y2 - y1)]
        pError = trackFace(info, w, pid, pError)
        cv2.putText(img, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)  # Display name
        # Speak out the name
        engine.say(f"I see {name}")
        engine.runAndWait()
    else:
        # Face not recognized, continue scanning for a face
        img, info = findFace(img)
        pError = trackFace(info, w, pid, pError)

    cv2.imshow("Output", img)
    
    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        me.land()
        break

cv2.destroyAllWindows()
