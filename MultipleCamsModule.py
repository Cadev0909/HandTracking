import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm

# Initialize HandDetector
detector = htm.handDetector()

# Print available cameras
index = 0
pTime = 0
temp = 0

while True:
    cap = cv2.VideoCapture(index)
    print("Camera No.: {} captured as {}".format(index, cap))
    if not cap.read()[0]:
        break
    #cap.release()
    index += 1

# Set up cameras
cap1 = cv2.VideoCapture(0)  # Camera 1
time.sleep(1)
cap2 = cv2.VideoCapture(1)  # Camera 2
time.sleep(1)
cap3 = cv2.VideoCapture(2)  # Camera 3
time.sleep(1)
cap4 = cv2.VideoCapture(3)  # Camera 4


while True:
    # Read frames from cameras
    success1, img1 = cap1.read()
    success2, img2 = cap2.read()
    success3, img3 = cap3.read()
    success4, img4 = cap4.read()

    # Print success values for debugging
    while temp == 0:
        print("Camera 1 success:", success1)
        print("Camera 2 success:", success2)
        print("Camera 3 success:", success3)
        print("Camera 4 success:", success4)
        temp = 1
        exit;

    # Check if frames were successfully read
    if not success1 or not success2 or not success3 or not success4:
        print("Failed to read frame(s). Exiting.")
        break

    # Change width and height of cameras
    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap3.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap3.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap4.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap4.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Find hands in the frames
    img1 = detector.findHands(img1, draw=False)
    img2 = detector.findHands(img2, draw=False)
    img3 = detector.findHands(img3, draw=False)
    img4 = detector.findHands(img4, draw=False)

    # Find hand positions
    lmList1 = detector.findPosition(img1, draw=True)
    lmList2 = detector.findPosition(img2, draw=True)
    lmList3 = detector.findPosition(img3, draw=True)
    lmList4 = detector.findPosition(img4, draw=False)

    # Process landmarks if available
    if len(lmList1) != 0:
        print("Camera 1 - Landmark 4:", lmList1[4])

    if len(lmList2) != 0:
        print("Camera 2 - Landmark 4:", lmList2[4])

    if len(lmList3) != 0:
        print("Camera 3 - Landmark 4:", lmList3[4])

    if len(lmList4) != 0:
        print("Camera 4 - Landmark 4:", lmList4[4])

    # Display FPS on the frames
    fps = 1 / (time.time() - pTime)
    pTime = time.time()
    cv2.putText(img1, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    #cv2.putText(img2, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    #cv2.putText(img3, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    # Display frames from all cameras
    cv2.imshow('Camera 1', img1)
    #cv2.imshow('Camera 2', img2)
    #cv2.imshow('Camera 3', img3)
    #cv2.imshow('Camera 4', img4)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera instances
cap1.release()
cap2.release()
cap3.release()
cap4.release()
cv2.destroyAllWindows()
