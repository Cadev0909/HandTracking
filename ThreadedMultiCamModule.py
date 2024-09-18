import cv2
import time
import threading
import HandTrackingModule as htm

# Function to process frames from a specific camera
def process_camera(i):
    cap = cv2.VideoCapture(i)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    pTime = time.time()  # Initialize pTime here
    while True:
        success, frame = cap.read()

        if success:
            frame = detector.findHands(frame, draw=False)
            lmList = detector.findPosition(frame, draw=True)

            if lmList:
                print(f"Camera {i} - Landmark 4:", lmList[4])  # Print landmark info

            fps = 1 / (time.time() - pTime)
            pTime = time.time()  # Update pTime here
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            cv2.imshow(f'Camera {i}', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

# Initialize HandDetector
detector = htm.handDetector()

# Find available camera indices
available_indices = []
for i in range(5):  # Try indices from 0 to 9
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera at index {i} is available.")
        available_indices.append(i)
        cap.release()
    else:
        print(f"No camera detected at index {i}.")

# Create and start threads for each available camera
threads = []
for idx in available_indices:
    thread = threading.Thread(target=process_camera, args=(idx,))
    threads.append(thread)
    thread.start()

# Wait for threads to complete
for thread in threads:
    thread.join()

cv2.destroyAllWindows()
