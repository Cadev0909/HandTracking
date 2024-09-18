import cv2
import time
import HandTrackingModule as htm

# Initialize HandDetector
detector = htm.handDetector()

# Adjust index (i) to filter through cams where variable in array is present
Num_cams = 4
pTime = 0
cap = [cv2.VideoCapture(i) for i in range(Num_cams)]
success = [False] * Num_cams
img = [None] * Num_cams
lmList = [None] * Num_cams

# Change width and height of cameras for better FPS
for i in range(Num_cams):
    cap[i].set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap[i].set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    for i in range(Num_cams):
        success[i], img[i] = cap[i].read()

        if img[i] is not None:
            img[i] = detector.findHands(img[i], draw=False)
            lmList[i] = detector.findPosition(img[i], draw=True)

            if lmList[i]:
                print(f"Camera {i+1} - Landmark 4:", lmList[i][4])

            fps = 1 / (time.time() - pTime)
            pTime = time.time()
            cv2.putText(img[i], f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            cv2.imshow(f'Camera {i+1}', img[i])

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

for i in range(Num_cams):
    cap[i].release()
cv2.destroyAllWindows()
