import cv2
import mediapipe as mp

# Initialize MediaPipe Hand
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# Finger tip landmarks
finger_tips = [8, 12, 16, 20]
thumb_tip = 4

# Start Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    finger_count = 0

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            lm = hand_landmarks.landmark

            # Thumb (left hand logic)
            if lm[thumb_tip].x > lm[thumb_tip - 1].x:
                finger_count += 1

            # Other 4 fingers
            for tip in finger_tips:
                if lm[tip].y < lm[tip - 2].y:
                    finger_count += 1

            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

    # Display result
    cv2.rectangle(frame, (20, 20), (250, 100), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"Hand Cricket: {finger_count}",
        (30, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Hand Cricket - Number Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
