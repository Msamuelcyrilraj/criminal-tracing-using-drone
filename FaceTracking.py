import cv2
import numpy as np
from djitellopy import Tello
import time

me = Tello()
me.connect()
print(me.get_battery())

me.streamon()
me.takeoff()
me.send_rc_control(0,0,28,0)
me.move_up(100)
time.sleep(2.2)

w , h = 360 ,240
fbRange = [6200,6800]
pid = [0.4 , 0.4 , 0]
pError = 0

def findFace(img):
    fb= 0
    faceCascade = cv2.CascadeClassifier("D:\MAJOR PROJECT\project\Resources\haarcascade_frontalface_default.xml")
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray,1.2,8)

    myFaceListC = []
    myFaceListArea = []

    for (x,y,w,h) in faces:
        print(x,y,w,h)
        cv2.rectangle(img,(x,y),(x + w ,y + h) , (0 ,0, 255) , 2)
        # cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        cx = x + w // 2
        #cx = x1 + (x2-x1) // 2
        cy = y + h // 2
        #cy = y1 + (y2-y1) // 2
        area = w * h
        #area = (x2-x1) * (y2-y1)
        cv2.circle(img, (cx,cy) , 5 , (0,255,0),cv2.FILLED)
        myFaceListC.append([cx, cy])
        myFaceListArea.append(area)

    if len(myFaceListArea) != 0:
        i = myFaceListArea.index(max(myFaceListArea))
        return img, [myFaceListC[i] ,myFaceListArea[i]]
        
        
    else:
        return img , [[0,0],0]    


def trackFace(info , w, pid , pError):
    area = info[1]
    x,y = info[0]
    fb=0
    print([x,y] , area)
    error = x - w//2
    speed = pid[0]*error + pid[1] *(error - pError)
    speed = int(np.clip(speed , -100, 100))
    if area >fbRange[0] and area < fbRange[1]:
        fb =0
    elif area > fbRange[1]:
        fb = -20
    elif area < fbRange[0] and area !=0:
        fb = 20

    if x ==0:
        speed = 0
        error = 0    

    print(speed,fb)

    me.send_rc_control(0,fb,0,speed) 
    return error          

# cap = cv2.VideoCapture(0)
while True:
    # _,img = cap.read()
    img = me.get_frame_read().frame
    img = cv2.resize(img,(w,h))
    img,info = findFace(img)
    pError = trackFace(info,w, pid , pError)
    cv2.imshow("Output",img)
    # cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        me.land()
        break