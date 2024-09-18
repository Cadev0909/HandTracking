import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm



pTime = 0
cTime = 0
# set camera
cap = cv2.VideoCapture(0)
detector = htm.handDetector()

while True:
    success, img = cap.read()
    img = detector.findHands(img, draw=False)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        print(lmList[4])  # 4 being the landmark point on hand

    cTime = time.time()
    fps = 1 / (cTime - pTime)  # make fps
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)  # adjust fps

    cv2.imshow('Image', img)
    cv2.waitKey(1)