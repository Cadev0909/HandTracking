import cv2
import mediapipe as mp
import time
import numpy as np

# Initialize HandDetector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Print available cameras
index = 0
pTime = 0
temp = 0
while True:
    cap = cv2.VideoCapture(index)
    print("Camera No.: {} captured as {}".format(index, cap))
    if not cap.read()[0]:
        break
    cap.release()
    index += 1

# Set up cameras
cap1 = cv2.VideoCapture(0)  # Camera 1
time.sleep(1)
cap2 = cv2.VideoCapture(1)  # Camera 2

# Calibration variables
calibration_frames = 10
calibration_count = 0
calibration_lm_list1 = []
calibration_lm_list2 = []
calibration_matrix = None

while True:
    # Read frames from cameras
    success1, img1 = cap1.read()
    success2, img2 = cap2.read()

    # Print success values for debugging
    while temp == 0:
        print("Camera 1 success:", success1)
        print("Camera 2 success:", success2)
        temp = 1

    # Check if frames were successfully read
    if not success1 or not success2:
        print("Failed to read frame(s). Exiting...")
        break

    # Perform calibration
    if calibration_count < calibration_frames:
        # Convert the BGR image to RGB
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        # Find hands in the frames
        results1 = hands.process(img1_rgb)
        results2 = hands.process(img2_rgb)

        # Find hand positions for calibration
        if results1.multi_hand_landmarks and results2.multi_hand_landmarks:
            landmarks1 = results1.multi_hand_landmarks[0].landmark[0]  # Use the first detected hand
            landmarks2 = results2.multi_hand_landmarks[0].landmark[0]

            calibration_lm_list1.append(landmarks1)
            calibration_lm_list2.append(landmarks2)
            calibration_count += 1

            cv2.putText(img1, f"Calibrating {calibration_count}/{calibration_frames}",
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Camera 1', img1)

            if calibration_count == calibration_frames:
                # Perform calibration
                calibration_lm_array1 = np.array(calibration_lm_list1)
                calibration_lm_array2 = np.array(calibration_lm_list2)

                # Check if enough points for calibration
                if calibration_lm_array1.shape[0] < 4 or calibration_lm_array2.shape[0] < 4:
                    print("Insufficient points for calibration. Please retry with more distinct hand movements.")
                    break

                _, rvec, tvec, inliers = cv2.solvePnPRansac(
                    calibration_lm_array2, calibration_lm_array1, np.eye(3), None
                )

                # Check if enough inliers after calibration
                if inliers is None or len(inliers) < 4:
                    print("Insufficient inliers after calibration. Please retry with more distinct hand movements.")
                    break

                # Convert rotation vector to rotation matrix
                rmat = cv2.Rodrigues(rvec)[0]

                # Create the transformation matrix
                calibration_matrix = np.hstack((rmat, tvec))

                print("Calibration complete.")

    else:
        # Convert the BGR image to RGB
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        # Find hands in the frames
        results1 = hands.process(img1_rgb)
        results2 = hands.process(img2_rgb)

        # Process landmarks if available
        if results1.multi_hand_landmarks and results2.multi_hand_landmarks:
            # Apply calibration transformation to landmarks2
            landmarks2 = results2.multi_hand_landmarks[0].landmark[0]
            transformed_landmarks2 = cv2.perspectiveTransform(
                np.array([landmarks2], dtype=np.float32), calibration_matrix
            )[0]

            print("Camera 1 - Landmark 4:", calibration_lm_list1[0])
            print("Camera 2 - Landmark 4 (calibrated):", transformed_landmarks2)

        # Display FPS on the frames
        fps = 1 / (time.time() - pTime)
        pTime = time.time()
        cv2.putText(img1, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        # Display frames from both cameras
        cv2.imshow('Camera 1', img1)
        cv2.imshow('Camera 2', img2)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release camera instances
cap1.release()
cap2.release()
cv2.destroyAllWindows()
