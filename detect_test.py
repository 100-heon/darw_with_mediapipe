import os
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import model


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.7)

model = model.CNNHandGestureModel()
model.load_state_dict(torch.load('cnn_hand_gesture_model_gray_deeper.pth', map_location=torch.device('cpu')), strict=False)
model.eval()


def clear_canvas():
    global canvas, prev_coords, recent_points
    canvas = np.zeros_like(canvas)
    recent_points = {0: deque(maxlen=5), 1: deque(maxlen=5)}  

def live_inference(webcam):
    global canvas, prev_coords, recent_points
    cap = cv2.VideoCapture(webcam)
    cap.set(cv2.CAP_PROP_FPS, 60)  

    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    canvas = None
    prev_coords = {0: (None, None), 1: (None, None)}
    recent_points = {0: deque(maxlen=3), 1: deque(maxlen=5)}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        if canvas is None or canvas.shape != frame.shape:
            canvas = np.zeros_like(frame)

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(image_rgb)

        if result.multi_hand_landmarks:
            labels = []

            for hand_index, hand_landmarks in enumerate(result.multi_hand_landmarks):
                joint = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark])
                base_x, base_y = joint[0]
                relative_joint = joint - [base_x, base_y]
                relative_joint = relative_joint.reshape(1, 1, 21, 2)

                input_tensor = torch.tensor(relative_joint, dtype=torch.float32)
                output = model(input_tensor)
                _, predicted = torch.max(output, 1)
                label = int(predicted.item())
                labels.append(label)

                if label == 1:
                    index_finger_tip = hand_landmarks.landmark[8]
                    h, w = gray_frame.shape
                    x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

                    prev_x, prev_y = prev_coords[hand_index]
                    if prev_x is not None and prev_y is not None:
                        cv2.line(canvas, (prev_x, prev_y), (x, y), (255, 255, 255), 8)

                    prev_coords[hand_index] = (x, y)
                    recent_points[hand_index].append((prev_x, prev_y, x, y)) 

                elif label == 5:

                    while recent_points[hand_index]:
                        x1, y1, x2, y2 = recent_points[hand_index].popleft()
                        if x1 is not None and y1 is not None and x2 is not None and y2 is not None:
                            cv2.line(canvas, (x1, y1), (x2, y2), (0, 0, 0), 8)

                    prev_coords[hand_index] = (None, None)

                cv2.putText(frame, f'Predicted: {label}', (10, 50 + hand_index * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if set(labels) == {0, 5}:
                clear_canvas()

        np.copyto(frame, canvas, where=(canvas != 0))

        cv2.imshow("Hand Gesture Recognition", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

webcam = 0
live_inference(webcam)
