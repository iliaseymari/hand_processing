import cv2
import mediapipe as mp
import time

# ------------------------------
# Configuration Parameters
# ------------------------------
MAX_NUM_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# Finger landmark indices for counting
FINGER_TIPS = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
FINGER_PIPS = [6, 10, 14, 18]
THUMB_TIP = 4
THUMB_IP = 3

# ------------------------------
# Helper Functions
# ------------------------------
def count_open_fingers(hand_landmarks, handedness_label):
    """
    Count number of open fingers for a single hand.
    hand_landmarks: landmarks list
    handedness_label: 'Left' or 'Right'
    """
    open_count = 0
    # For 4 fingers (excluding thumb)
    for tip, pip in zip(FINGER_TIPS, FINGER_PIPS):
        # If tip is above pip in y-axis (smaller y value) => finger open
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
            open_count += 1
    # Thumb logic differs by handedness (x-axis)
    if handedness_label == 'Right':
        if hand_landmarks.landmark[THUMB_TIP].x < hand_landmarks.landmark[THUMB_IP].x:
            open_count += 1
    else:
        if hand_landmarks.landmark[THUMB_TIP].x > hand_landmarks.landmark[THUMB_IP].x:
            open_count += 1
    return open_count

# ------------------------------
# Main Application
# ------------------------------
def main():
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return

    # Optionally mirror feed for natural interaction
    MIRROR_FEED = True

    # Frame rate calculation
    prev_time = 0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Cannot read frame")
                break

            # Mirror image if desired
            if MIRROR_FEED:
                frame = cv2.flip(frame, 1)

            # Convert to RGB for processing
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_rgb.flags.writeable = False
            results = hands.process(img_rgb)

            # Restore writeable flag and convert back to BGR for OpenCV
            img_rgb.flags.writeable = True
            frame = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            total_hands = 0
            total_fingers = 0

            # If hands detected
            if results.multi_hand_landmarks and results.multi_handedness:
                total_hands = len(results.multi_hand_landmarks)
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                      results.multi_handedness):
                    # Draw landmarks & connections
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Count open fingers
                    hand_label = handedness.classification[0].label  # 'Left' or 'Right'
                    open_fingers = count_open_fingers(hand_landmarks, hand_label)
                    total_fingers += open_fingers

            # Overlay counts
            overlay_text = f"Hands: {total_hands}  |  Fingers: {total_fingers}"
            cv2.putText(frame, overlay_text,
                        org=(10, 30),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        color=(0, 255, 0),
                        thickness=2)

            # Calculate and show FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time else 0
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            # Display frame
            cv2.imshow('Hand & Finger Counter', frame)

            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
