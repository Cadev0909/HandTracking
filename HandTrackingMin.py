import cv2
import mediapipe as mp
import time

#set camera
cap = cv2.VideoCapture(0)

#write class for hand detection
mpHands = mp.solutions.hands
hands = mpHands.Hands() #keeping default values, can change detection if pass values ex: only 2 hand currently tracked
mpDraw = mp.solutions.drawing_utils #draws the hand movement landmarks

pTime = 0
cTime = 0

while True:
    success, img = cap.read()

    #convert image into rgb because class only uses rgb images
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #process the frame, now need to extract data
    results = hands.process(imgRGB)

    #print hand tracking
    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLMS in results.multi_hand_landmarks: #handLMS is each individual hand
            for id, lm in enumerate(handLMS.landmark):
                #print(id,lm) #print landmarks
                h, w, c = img.shape #gives width and height of img
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id, cx, cy)

                if id == 0: #palm
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED) #draw on particular landmark

                mpDraw.draw_landmarks(img, handLMS, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1 / (cTime - pTime) #make fps
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_SIMPLEX,3, (255,0,255), 3) #adjust fps

    cv2.imshow('Image', img)
    cv2.waitKey(1)