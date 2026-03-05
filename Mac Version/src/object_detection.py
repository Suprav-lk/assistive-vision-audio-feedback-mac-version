import threading
from ultralytics import YOLO
import cv2
import os
import time


# -------------------------------
# 1. LOAD YOLO MODEL
# -------------------------------

# Load lightweight YOLOv8 model
model = YOLO("yolov8n.pt")

# Move model to Apple Silicon GPU (MPS acceleration)
model.to("mps")


# -------------------------------
# 2. INITIALIZE CAMERA
# -------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access camera")
    exit()

# Reduce camera resolution to improve performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

print("Assistive Vision started. Press 'q' to quit.")


# -------------------------------
# 3. PRIORITY OBJECT LIST
# -------------------------------
# Only these objects will trigger audio feedback

important_objects = [
    "person",
    "car",
    "bus",
    "truck",
    "bicycle",
    "motorcycle",
    "bottle"
]


# -------------------------------
# 4. SPEECH CONTROL VARIABLES
# -------------------------------

last_spoken_time = 0      # Stores last time system spoke
speech_delay = 8          # Minimum seconds between announcements
frame_count = 0           # Used for frame skipping
last_announcements = set() # Stores previously announced objects to detect environmental changes


# -------------------------------
# 5. SPEECH FUNCTION (Threaded)
# -------------------------------

def speak(text):
    # Uses macOS built-in speech engine
    os.system(f"say {text}")


# -------------------------------
# 6. MAIN LOOP
# -------------------------------

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # ----------------------------------
    # PERFORMANCE OPTIMIZATION
    # ----------------------------------
    # Skip frames to reduce CPU/GPU load
    if frame_count % 3 != 0:
        cv2.imshow("Assistive Vision - Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue


    # ----------------------------------
    # 7. RUN OBJECT DETECTION
    # ----------------------------------

    results = model(frame, conf=0.3, stream=True)

    # This list will store final spoken messages
    detected_announcements = []

    # Get frame dimensions
    height, width, _ = frame.shape
    frame_area = width * height   # Used for distance estimation


    # ----------------------------------
    # 8. PROCESS DETECTION RESULTS
    # ----------------------------------

    for r in results:

        # Draw bounding boxes on frame
        annotated_frame = r.plot()

        for box in r.boxes:

            # Get detected object class
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            # Ignore non-priority objects
            if class_name not in important_objects:
                continue


            # ----------------------------------
            # 9. DIRECTION ESTIMATION
            # ----------------------------------

            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]

            # Compute horizontal center of object
            center_x = (x1 + x2) / 2

            # Divide screen into 3 vertical zones
            if center_x < width / 3:
                direction = "on your left"
            elif center_x < (2 * width / 3):
                direction = "in front of you"
            else:
                direction = "on your right"


            # ----------------------------------
            # 10. DISTANCE ESTIMATION
            # ----------------------------------

            # Calculate bounding box size
            box_width = x2 - x1
            box_height = y2 - y1
            box_area = box_width * box_height

            # Compare object size relative to full frame
            relative_size = box_area / frame_area

            # Categorize distance based on size ratio
            if relative_size > 0.15:
                distance = "very close"
            elif relative_size > 0.05:
                distance = "nearby"
            else:
                distance = "far"


            # Combine object + distance + direction
            detected_announcements.append(
                f"{class_name} {distance} {direction}"
            )


    # ----------------------------------
    # 11. SPEECH BLOCK
    # ----------------------------------
    # This is the part that controls when audio happens

    # ----------------------------------
    # 11. EVENT-BASED SPEECH BLOCK
    # ----------------------------------

    current_set = set(detected_announcements)

    # Speak only if environment changed
    if current_set != last_announcements and current_set:

        announcement = ", ".join(current_set)

        threading.Thread(
            target=speak,
            args=(announcement,)
        ).start()

        print("Speaking:", announcement)

        # Update memory of last environment state
        last_announcements = current_set

    # ----------------------------------
    # 12. DISPLAY FRAME
    # ----------------------------------

    cv2.imshow("Assistive Vision - Object Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# -------------------------------
# 13. CLEANUP
# -------------------------------

cap.release()
cv2.destroyAllWindows()