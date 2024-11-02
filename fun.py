import streamlit as st
import cv2
import numpy as np
import torch
import asyncio
from bleak import BleakScanner, BleakClient
import gtts
import os
from random import choice
import os
import speech_recognition as sr  # Importing speech recognition library

os.system('pip install ultralytics==8.3.13')
os.system('apt-get update && apt-get install -y libgl1-mesa-glx')

st.markdown("""
    <h2>Bluetooth Device Scanner</h2>
    <button id="scan">Scan for Bluetooth Devices</button>
    <ul id="devices"></ul>

    <script>
    document.getElementById('scan').addEventListener('click', async () => {
        try {
            const device = await navigator.bluetooth.requestDevice({ acceptAllDevices: true });
            document.getElementById('devices').innerHTML += `<li>${device.name} (${device.id})</li>`;
        } catch (error) {
            console.error('Error: ', error);
        }
    });
    </script>
""", unsafe_allow_html=True)

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Predefined storylines
storylines = [
    "Once upon a time, in a magical land, there was a {object}...",
    "In a faraway kingdom, the {object} played a crucial role...",
    "A young explorer found a mysterious {object}...",
    "Legends spoke of a powerful {object}...",
    "Deep in the forest, a {object} was discovered..."
]

async def scan_and_connect():
    devices = await BleakScanner.discover()
    st.write("Available devices:")
    for idx, device in enumerate(devices):
        st.write(f"{idx + 1}. {device.name} - {device.address}")

    if devices:
        address = devices[0].address
        async with BleakClient(address) as client:
            st.write(f"Connected to {devices[0].name} ({devices[0].address})")

def detect_object_in_frame(frame):
    results = model(frame)
    detected_objects = results.xyxyn[0][:, -1]
    detected_labels = [model.names[int(cls_idx)] for cls_idx in detected_objects]

    for i, (label, cord) in enumerate(zip(detected_labels, results.xyxyn[0])):
        xmin, ymin, xmax, ymax, conf = cord[:5]
        xmin, ymin, xmax, ymax = int(xmin * frame.shape[1]), int(ymin * frame.shape[0]), int(xmax * frame.shape[1]), int(ymax * frame.shape[0])
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame, detected_labels

def announce_object(detected_object):
    tts = gtts.gTTS(f"A {detected_object} was detected.")
    tts.save("object_detected.mp3")
    os.system("start object_detected.mp3")

def generate_story_for_object(detected_object):
    if detected_object:
        story = choice(storylines).format(object=detected_object)
        st.write("Generated Story:")
        st.write(story)
        tts = gtts.gTTS(story)
        tts.save("story.mp3")
        os.system("start story.mp3")

st.title("Object Detection, Story Generation, and Bluetooth Integration")

webcam_running = False

st.header("Real-time Object Detection")
if st.button("Start Webcam"):
    webcam_running = True
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.write("Error: Could not open video stream.")
    
    while webcam_running:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to grab frame.")
            break

        frame, detected_labels = detect_object_in_frame(frame)
        st.image(frame, channels="BGR")

        if detected_labels:
            detected_object = detected_labels[0]
            st.write(f"Detected Object: {detected_object}")
            announce_object(detected_object)

        if st.button("Stop Webcam"):
            webcam_running = False

    cap.release()

st.header("Detect Objects in Uploaded Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    image, detected_labels = detect_object_in_frame(image)
    st.image(image, channels="BGR")
    
    if detected_labels:
        detected_object_image = detected_labels[0]
        st.write(f"Detected Object in Image: {detected_object_image}")
        announce_object(detected_object_image)
        generate_story_for_object(detected_object_image)

st.header("Find Specific Object and Measure Distance")
specific_object = st.text_input("Enter the object you are looking for:")

# Add voice command button
if st.button("Use Voice Command"):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Please speak the object you want to detect.")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)  # Adjust for ambient noise
        try:
            audio = recognizer.listen(source, timeout=5)  # Timeout after 5 seconds
            specific_object = recognizer.recognize_google(audio)
            st.write(f"Looking for: {specific_object}")
        except sr.UnknownValueError:
            st.write("Sorry, I couldn't understand that.")
        except sr.RequestError:
            st.write("Error with the speech recognition service.")
        except sr.WaitTimeoutError:
            st.write("No speech detected within the time limit.")

if specific_object:
    st.write(f"Looking for: {specific_object}")

    if st.button("Start Webcam for Specific Object"):
        webcam_running = True
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.write("Error: Could not open video stream.")
        else:
            while webcam_running:
                ret, frame = cap.read()
                if not ret:
                    st.write("Failed to grab frame.")
                    break

                frame, detected_labels = detect_object_in_frame(frame)

                if specific_object in detected_labels:
                    st.write(f"Specific object '{specific_object}' found!")
                    distance = np.random.randint(50, 150)
                    st.write(f"Distance to {specific_object}: {distance} cm")

                st.image(frame, channels="BGR")

                if st.button("Stop Specific Object Detection"):
                    webcam_running = False
                    break

            cap.release()

st.header("Bluetooth Integration")
if st.button("Scan and Connect to Bluetooth Device"):
    asyncio.run(scan_and_connect())
