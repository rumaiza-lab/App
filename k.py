import streamlit as st
import cv2
import numpy as np
import torch
import asyncio
from bleak import BleakScanner, BleakClient
import gtts
import os
from random import choice
import streamlit as st
import os
os.system('pip install ultralytics==8.3.13')
os.system('apt-get update && apt-get install -y libgl1-mesa-glx')

# Inject HTML and JavaScript into Streamlit app
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

# Main layout of the app
st.title("Object Detection, Story Generation, and Bluetooth Integration")
st.write("Click the button above to scan for nearby Bluetooth devices using the Web Bluetooth API.")


# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Predefined storylines with at least one full paragraph
storylines = [
    "Once upon a time, in a magical land, there was a {object} that everyone admired. The object had special powers, and people from all over the kingdom came to witness its glory. One day, a brave adventurer decided to seek the {object} for his quest. On his journey, he met friends and foes alike, but the {object} remained at the center of his quest. The adventurer soon learned that the true power of the {object} was not its magical properties, but the lessons it taught him along the way.",
    "In a faraway kingdom, the {object} played a crucial role in a great adventure. A young hero was chosen by the king to retrieve the {object} from a distant land. Along the way, the hero encountered numerous challenges, but with each challenge, he grew stronger. The {object} was said to hold the power to bring peace to the kingdom, and the hero was determined to return it to the king. However, when he finally found the {object}, he realized that the real treasure was the friends he made on his journey.",
    "A young explorer found a mysterious {object} that changed the course of history. The {object} was hidden deep in an ancient temple, guarded by traps and puzzles. With wit and bravery, the explorer solved the mysteries of the temple and uncovered the {object}. But as soon as the {object} was taken, the temple began to collapse. The explorer narrowly escaped, but not without the realization that the {object} held a key to the future of humanity.",
    "Legends spoke of a powerful {object} that could unlock untold mysteries. Many tried to find it, but only a few succeeded. Among those few was a scholar who devoted his life to studying the {object}. After years of research, he finally deciphered its true meaning. The {object}, it turns out, was not a tool for power, but a guide to understanding the world around us. The scholar's discovery changed the way people viewed the {object}, and it became a symbol of knowledge and enlightenment.",
    "Deep in the forest, a {object} was discovered, leading to an unexpected journey. The {object} was no ordinary artifact—it had the power to communicate with nature. Those who possessed it could speak to animals and trees, understanding the secrets of the forest. A group of travelers stumbled upon the {object} while lost in the woods. With its help, they navigated their way back home, forever changed by the experience."
]

# Bluetooth scanner
async def scan_and_connect():
    devices = await BleakScanner.discover()
    st.write("Available devices:")
    for idx, device in enumerate(devices):
        st.write(f"{idx + 1}. {device.name} - {device.address}")

    # Example: Connect to the first available device
    if devices:
        address = devices[0].address
        async with BleakClient(address) as client:
            st.write(f"Connected to {devices[0].name} ({devices[0].address})")

# Detect object in a frame
def detect_object_in_frame(frame):
    results = model(frame)
    detected_objects = results.xyxyn[0][:, -1]  # Extracting class indexes
    detected_labels = [model.names[int(cls_idx)] for cls_idx in detected_objects]  # Class names
    
    # Draw bounding boxes on the image and annotate detected objects
    for i, (label, cord) in enumerate(zip(detected_labels, results.xyxyn[0])):
        xmin, ymin, xmax, ymax, conf = cord[:5]
        xmin, ymin, xmax, ymax = int(xmin * frame.shape[1]), int(ymin * frame.shape[0]), int(xmax * frame.shape[1]), int(ymax * frame.shape[0])
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame, detected_labels

# Announce detected object using text-to-speech
def announce_object(detected_object):
    tts = gtts.gTTS(f"A {detected_object} was detected.")
    tts.save("object_detected.mp3")
    os.system("start object_detected.mp3")

# Generate a story for the detected object
def generate_story_for_object(detected_object):
    if detected_object:
        story = choice(storylines).format(object=detected_object)
        st.write("Generated Story:")
        st.write(story)

        # Convert story to speech
        tts = gtts.gTTS(story)
        tts.save("story.mp3")
        os.system("start story.mp3")

# Main app layout
st.title("Object Detection, Story Generation, and Bluetooth Integration")

# Initialize webcam variable
webcam_running = False

# Real-time object detection
st.header("Real-time Object Detection")
if st.button("Start Webcam"):
    webcam_running = True
    cap = cv2.VideoCapture(0)  # Use 0 for default webcam
    
    if not cap.isOpened():
        st.write("Error: Could not open video stream.")
    
    # Loop to display webcam feed
    while webcam_running:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to grab frame.")
            break

        # Detect objects in the current frame
        frame, detected_labels = detect_object_in_frame(frame)

        # Display the frame with object detections
        st.image(frame, channels="BGR")

        # Announce detected object without generating a story
        if detected_labels:
            detected_object = detected_labels[0]
            st.write(f"Detected Object: {detected_object}")
            announce_object(detected_object)

        # Stop the camera
        if st.button("Stop Webcam"):
            webcam_running = False

    cap.release()

# Upload image for object detection
st.header("Detect Objects in Uploaded Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Detect objects in uploaded image
    image, detected_labels = detect_object_in_frame(image)
    
    # Display the uploaded image and detections
    st.image(image, channels="BGR")
    
    if detected_labels:
        detected_object_image = detected_labels[0]
        st.write(f"Detected Object in Image: {detected_object_image}")
        announce_object(detected_object_image)
        generate_story_for_object(detected_object_image)

# Find specific object
st.header("Find Specific Object and Measure Distance")
specific_object = st.text_input("Enter the object you are looking for:")
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

                # Detect objects in the frame
                frame, detected_labels = detect_object_in_frame(frame)

                # Check for specific object and calculate distance (using placeholder logic)
                if specific_object in detected_labels:
                    st.write(f"Specific object '{specific_object}' found!")
                    distance = np.random.randint(50, 150)  # Placeholder distance calculation
                    st.write(f"Distance to {specific_object}: {distance} cm")

                # Show the webcam feed
                st.image(frame, channels="BGR")

                # Stop the camera
                if st.button("Stop Specific Object Detection"):
                    webcam_running = False
                    break

            cap.release()

# Bluetooth option
st.header("Bluetooth Integration")
if st.button("Scan and Connect to Bluetooth Device"):
    asyncio.run(scan_and_connect())
