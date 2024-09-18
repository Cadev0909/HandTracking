import cv2
import mediapipe as mp
import time
import sys

# Initialize MediaPipe Hands (can be changed for road/lane tracking)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# Function to process video
def process_video(video_source):
    cap = cv2.VideoCapture(video_source)

    pTime = 0
    cTime = 0

    # List to store the tracked points
    path_points = []

    while True:
        success, img = cap.read()
        if not success:
            break

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for handLMS in results.multi_hand_landmarks:
                for id, lm in enumerate(handLMS.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)

                    if id == 0:  # Track a specific point (like the palm)
                        path_points.append((cx, cy))
                        cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

                    mpDraw.draw_landmarks(img, handLMS, mpHands.HAND_CONNECTIONS)

        # Draw the path using the stored points
        for i in range(1, len(path_points)):
            cv2.line(img, path_points[i - 1], path_points[i], (0, 255, 0), 3)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)

        cv2.imshow('Image', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # If a video file path is provided as an argument, process the video file
        video_path = sys.argv[1]
        process_video(video_path)
    else:
        # Otherwise, use the real-time video from the camera
        process_video(0)
