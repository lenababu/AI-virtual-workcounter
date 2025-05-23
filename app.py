import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import base64
import time

# Load and convert image to base64 for CSS background
def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img_base64 = get_base64_image("gym.jpg")

# Custom CSS Styling
st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] {{
    background: linear-gradient(135deg, #000000, #0d47a1);
    background-image: url("data:image/jpg;base64,{img_base64}");
    background-size: cover;
    color: #e0f7fa;
    min-height: 100vh;
}}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 style="color: red;">🏋️ AI Virtual Fitness Coach</h1>', unsafe_allow_html=True)

# Workout and options
workout = st.selectbox("Select your workout", ("Squats", "Biceps Curl", "Push-up"))
show_skeleton = st.checkbox("Show Skeleton Overlay", value=True)
show_fps = st.checkbox("Show FPS", value=True)
st.write(f"Selected workout: {workout}")

# Initialize MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Session State
for k in ["counter_squats", "counter_biceps", "counter_pushup", "stage", "prev_workout", "run"]:
    if k not in st.session_state:
        st.session_state[k] = 0 if 'counter' in k else None

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

# Count Functions
def count_squats(landmarks, image):
    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
           landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
             landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    angle = calculate_angle(hip, knee, ankle)

    if show_skeleton:
        cv2.putText(image, f"Angle: {int(angle)}", tuple(np.multiply(knee, [640, 480]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

    if angle > 165:
        st.session_state.stage = "Up"
    if angle < 100 and st.session_state.stage == "Up":
        st.session_state.stage = "Down"
        st.session_state.counter_squats += 1

def count_biceps_curl(landmarks, image):
    shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
             landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
    wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
             landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
    angle = calculate_angle(shoulder, elbow, wrist)

    if show_skeleton:
        cv2.putText(image, f"Angle: {int(angle)}", tuple(np.multiply(elbow, [640, 480]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

    if angle > 160:
        st.session_state.stage = "Down"
    if angle < 40 and st.session_state.stage == "Down":
        st.session_state.stage = "Up"
        st.session_state.counter_biceps += 1

def count_pushup(landmarks, image):
    shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
             landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
    wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
             landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
    angle = calculate_angle(shoulder, elbow, wrist)

    if show_skeleton:
        cv2.putText(image, f"Angle: {int(angle)}", tuple(np.multiply(elbow, [640, 480]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

    if angle > 165:
        st.session_state.stage = "Up"
    if angle < 100 and st.session_state.stage == "Up":
        st.session_state.stage = "Down"
        st.session_state.counter_pushup += 1

# Webcam buttons
start = st.button("Start Webcam")
stop = st.button("Stop Webcam")
frame_window = st.empty()

if start:
    st.session_state.run = True
if stop:
    st.session_state.run = False

# Webcam Loop
if st.session_state.run:
    cap = cv2.VideoCapture(0)
    prev_time = time.time()
    while st.session_state.run:
        ret, frame = cap.read()
        if not ret:
            st.warning("Could not read from webcam.")
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Reset state if workout changes
            if workout != st.session_state.prev_workout:
                st.session_state.stage = None
                st.session_state.prev_workout = workout

            if workout == "Squats":
                count_squats(landmarks, image)
                count = st.session_state.counter_squats
            elif workout == "Biceps Curl":
                count_biceps_curl(landmarks, image)
                count = st.session_state.counter_biceps
            elif workout == "Push-up":
                count_pushup(landmarks, image)
                count = st.session_state.counter_pushup

            if show_skeleton:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        else:
            count = 0
            cv2.putText(image, "No Pose Detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        # Rep Counter Display
        cv2.rectangle(image, (0, 0), (300, 90), (245, 117, 16), -1)
        cv2.putText(image, f'REPS: {count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(image, f'STAGE: {st.session_state.stage}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        if show_fps:
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            cv2.putText(image, f'FPS: {int(fps)}', (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        frame_window.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    cap.release()

# Final Summary
if st.button("Show Final Workout Counts"):
    st.subheader("🏁 Workout Summary")
    st.write(f"Squats done: {st.session_state.counter_squats}")
    st.write(f"Biceps Curls done: {st.session_state.counter_biceps}")
    st.write(f"Push-ups done: {st.session_state.counter_pushup}")
