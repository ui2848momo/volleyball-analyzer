import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tempfile
import os
from datetime import datetime
import math

st.set_page_config(layout="wide")
st.title("🏐 Volleyball Spike Analyzer (with Elbow Feedback)")

# Save directory
save_dir = "captures"
os.makedirs(save_dir, exist_ok=True)

# Mediapipe 초기화
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# HSV 슬라이더
st.sidebar.header("🎨 Ball HSV Range (Pink)")
h_min = st.sidebar.slider("H Min", 0, 179, 169)
s_min = st.sidebar.slider("S Min", 0, 255, 101)
v_min = st.sidebar.slider("V Min", 0, 255, 78)
h_max = st.sidebar.slider("H Max", 0, 179, 179)
s_max = st.sidebar.slider("S Max", 0, 255, 255)
v_max = st.sidebar.slider("V Max", 0, 255, 255)

lower_hsv = np.array([h_min, s_min, v_min])
upper_hsv = np.array([h_max, s_max, v_max])

# 웹캠 열기
cap = cv2.VideoCapture(0)
hit_cooldown = 0

st_frame = st.empty()

def calculate_angle(a, b, c):
    """Angle between three points: a (shoulder), b (elbow), c (wrist)"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ab = a - b
    cb = c - b
    angle = np.arccos(np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb)))
    return np.degrees(angle)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ball_center = None
    ball_radius = 0

    if contours:
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        if radius > 10:
            ball_center = (int(x), int(y))
            ball_radius = int(radius)
            cv2.circle(frame, ball_center, ball_radius, (0, 0, 255), 2)

    # Mediapipe 처리
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(rgb)
    pose_results = pose.process(rgb)

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            # index_finger_tip = landmark 8
            finger_tip = hand_landmarks.landmark[8]
            h, w, _ = frame.shape
            cx, cy = int(finger_tip.x * w), int(finger_tip.y * h)
            cv2.circle(frame, (cx, cy), 8, (255, 0, 0), -1)

            if ball_center:
                dist = np.linalg.norm(np.array(ball_center) - np.array([cx, cy]))
                if dist < ball_radius + 15:  # 충돌 조건
                    now = datetime.now().strftime('%Y%m%d_%H%M%S')
                    feedback = "No feedback"
                    angle_text = ""

                    if pose_results.pose_landmarks:
                        landmarks = pose_results.pose_landmarks.landmark
                        try:
                            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w,
                                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h]
                            elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * w,
                                     landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * h]
                            wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w,
                                     landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h]
                            angle = calculate_angle(shoulder, elbow, wrist)
                            angle_text = f"Elbow Angle: {int(angle)}°"

                            if angle <= 150:
                                feedback = "Straighten your elbow when hitting the ball"
                            else:
                                feedback = "Great form!"
                        except:
                            feedback = "Pose landmark missing"

                    # 피드백 표시
                    cv2.putText(frame, feedback, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
                    if angle_text:
                        cv2.putText(frame, angle_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 150, 255), 2)

                    filename = os.path.join(save_dir, f"hit_{now}.jpg")
                    cv2.imwrite(filename, frame)
                    st.image(frame, caption=f"✅ Hit detected and saved as {filename}", channels="BGR")
                    st.success(feedback)

    if pose_results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    st_frame.image(frame, channels="BGR")
    
    if hit_cooldown > 0:
        hit_cooldown -= 1

cap.release()
