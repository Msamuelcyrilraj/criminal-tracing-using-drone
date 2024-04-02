import cv2
import numpy as np
import face_recognition
import os
from djitellopy import Tello
import time

me = Tello()
me.connect()
print(me.get_battery())



w , h = 360 ,240
fbRange = [6500,6800]
pid = [0.4 , 0.4 , 0]
pError = 0

path = 'Images'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cls in myList:
    curImg = cv2.imread(f'{path}/{cls}')
    images.append(curImg)
    classNames.append(os.path.splitext(cls)[0])
print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)
print('Encoding Complete')

me.streamon()
me.takeoff()
me.send_rc_control(0,0,28,0)
me.move_up(100)
time.sleep(2.2)


def trackFace(info , w, pid , pError):
    area = info[1]
    x,y = info[0]
    # x=x+10.1
    fb=0

    error = x - w//2
    print(error)
    speed = pid[0]*error + pid[1] *(error - pError)
    speed = int(np.clip(speed , -100, 100))
    if area >fbRange[0] and area < fbRange[1]:
        fb =0
    elif area > fbRange[1]:
        fb = -50
    elif area < fbRange[0] and area !=0:
        fb = 50

    if x ==0:
        speed = 0
        error = 0

    print(speed,fb)

    me.send_rc_control(0,fb,0,speed)
    return error






# cap = cv2.VideoCapture(0)

while True:
    # success, img = cap.read()
    img = me.get_frame_read().frame
    imgS = cv2.resize(img, (w, h))

    # imgS = cv2.resize(img,(w,h),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
    info = []
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 = faceLoc
            # y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            y1= y1 - 18
            x2 = x2 + 10
            cx = x1 + (x2-x1) // 2
            # cx =cx + 40
            cy = y1 + (y2-y1) // 2
            area = (x2-x1) * (y2-y1)

            print([x1,y1] , [(x2-x1) , (y2-y1)])
            info.append([cx,cy])
            info.append(area)
            cv2.circle(imgS, (cx,cy) , 5 , (0,255,0),cv2.FILLED)

            cv2.rectangle(imgS,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(imgS,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(imgS,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)

    if(len(info) == 0):
        info.append([0,0])
        info.append(0)

    print(info)
    # cv2.imshow('WebCam', img)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
    pError = trackFace(info,w, pid , pError)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_RGB2BGR)
    cv2.imshow("Output",imgS)
    # cv2.waitKey(1)Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        me.land()
        break








