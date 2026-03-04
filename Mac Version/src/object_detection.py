from ultralytics import YOLO
import cv2
import os
import time

# Load YOLO model
model = YOLO("yolov8n.pt")

# Use Apple Silicon GPU (M2)
model.to("mps")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access camera")
    exit()

# Reduce resolution for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Real-time object detection started. Press 'q' to quit.")

last_spoken_time = 0
speech_delay = 3  # seconds between announcements
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Skip every other frame for smoother video
    if frame_count % 2 != 0:
        cv2.imshow("Assistive Vision - Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Run detection
    results = model(frame, conf=0.4)

    detected_objects = []

    for r in results:
        annotated_frame = r.plot()
        for box in r.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            detected_objects.append(class_name)

    current_time = time.time()

    # Speak only if enough time has passed
    if detected_objects and (current_time - last_spoken_time > speech_delay):
        unique_objects = list(set(detected_objects))
        announcement = ", ".join(unique_objects)

        os.system(f"say {announcement} detected")
        print("Speaking:", announcement)

        last_spoken_time = current_time

    cv2.imshow("Assistive Vision - Object Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()