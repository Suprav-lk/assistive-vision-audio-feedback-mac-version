import threading
from ultralytics import YOLO
import cv2
import os
import time

# Load YOLO model
model = YOLO("yolov8n.pt")
model.to("mps")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access camera")
    exit()

# Reduce resolution for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

print("Assistive Vision started. Press 'q' to quit.")

# PRIORITY OBJECTS ONLY
important_objects = [
    "person",
    "car",
    "bus",
    "truck",
    "bicycle",
    "motorcycle"
]

last_spoken_time = 0
speech_delay = 5
frame_count = 0

def speak(text):
    os.system(f"say {text}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Skip frames for smoother video
    if frame_count % 3 != 0:
        cv2.imshow("Assistive Vision - Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    results = model(frame, conf=0.3, stream=True)

    detected_announcements = []

    height, width, _ = frame.shape

    for r in results:
        annotated_frame = r.plot()

        for box in r.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            # Only care about important objects
            if class_name not in important_objects:
                continue

            x1, y1, x2, y2 = box.xyxy[0]
            center_x = (x1 + x2) / 2

            # Divide screen into 3 vertical zones
            if center_x < width / 3:
                direction = "on your left"
            elif center_x < (2 * width / 3):
                direction = "in front of you"
            else:
                direction = "on your right"

            detected_announcements.append(f"{class_name} {direction}")

    current_time = time.time()

    if detected_announcements and (current_time - last_spoken_time > speech_delay):

        unique_announcements = list(set(detected_announcements))
        announcement = ", ".join(unique_announcements)

        threading.Thread(
            target=speak,
            args=(announcement,)
        ).start()

        print("Speaking:", announcement)

        last_spoken_time = current_time

    cv2.imshow("Assistive Vision - Object Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()