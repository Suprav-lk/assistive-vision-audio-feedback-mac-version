import threading
from ultralytics import YOLO
import cv2
import os
import time

# DeepSORT tracker for stable multi-person tracking
# It assigns an ID to each detected person and keeps track of them
# across frames, preventing detection "flickering"
from deep_sort_realtime.deepsort_tracker import DeepSort

# ==================================================
# 1. LOAD YOLO MODEL
# ==================================================

# Load lightweight YOLOv8 model (good for real-time use)
model = YOLO("yolov8n.pt")

# Use Apple Silicon GPU acceleration (MPS)
model.to("mps")


# ==================================================
# PERSON TRACKER (Stable multi-person tracking)
# ==================================================

tracker = DeepSort(
    max_age=30,      # frames before a track disappears
    n_init=3,        # frames required to confirm tracking
    max_cosine_distance=0.4
)

# ==================================================
# 2. INITIALIZE CAMERA
# ==================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access camera")
    exit()

# Lower resolution = smoother performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

print("Assistive Vision started. Press 'q' to quit.")


# ==================================================
# 3. PRIORITY OBJECTS
# ==================================================
# Only these objects will trigger voice feedback

important_objects = [
    "person",
    "car",
    "bus",
    "truck",
    "bicycle",
    "motorcycle",
    "bottle"
]


# ==================================================
# 4. SYSTEM STATE VARIABLES
# ==================================================

frame_count = 0

# Stores the last spoken environmental state
last_announcements = set()

# Tracks if emergency warning is currently active
emergency_active = False

# Minimum time between speech announcements
speech_cooldown = 2.5   # seconds

# Timestamp of last speech
last_speech_time = 0

# ==================================================
# 5. SPEECH FUNCTION (Non-blocking)
# ==================================================
# Runs speech in a separate thread so video doesn't freeze

def speak(text):
    os.system(f"say {text}")


# ==================================================
# 6. MAIN LOOP
# ==================================================

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # ----------------------------------------------
    # PERFORMANCE OPTIMIZATION
    # Skip frames to reduce processing load
    # ----------------------------------------------
    if frame_count % 3 != 0:
        cv2.imshow("Assistive Vision - Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue


    # ----------------------------------------------
    # RUN OBJECT DETECTION
    # ----------------------------------------------
    results = model(frame, conf=0.3, stream=True)

    # Will store final messages to speak
    detected_announcements = []

    # Get frame size (used for distance estimation)
    height, width, _ = frame.shape
    frame_area = width * height


    # ==================================================
    # PROCESS EACH DETECTED OBJECT
    # ==================================================

    for r in results:

        # Draw bounding boxes on screen
        annotated_frame = r.plot()
        
        # --------------------------------------------------
        # LIST OF DETECTIONS FOR TRACKER
        # --------------------------------------------------
        """DeepSORT requires detections in a specific format:
            [bounding_box, confidence_score, class_name]
            We collect all person detections here before sending
            them to the tracker."""

        tracker_detections = []

        for box in r.boxes:

            # Get object name from YOLO class ID
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            
            # --------------------------------------------------
            # PREPARE PERSON DETECTIONS FOR TRACKING
            # --------------------------------------------------
            """We only track people because they are the most
                important dynamic obstacle for a visually impaired user."""
                        
            if class_name == "person":

                x1, y1, x2, y2 = box.xyxy[0]

                w = x2 - x1
                h = y2 - y1

                confidence = float(box.conf[0])

                tracker_detections.append(
                    ([int(x1), int(y1), int(w), int(h)], confidence, "person")
                )

            # Ignore objects that are not important
            if class_name not in important_objects:
                continue


            # ------------------------------------------
            # DIRECTION ESTIMATION
            # ------------------------------------------

            # Bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]

            # Horizontal center of object
            center_x = (x1 + x2) / 2

            if center_x < width / 3:
                direction = "on your left"
            elif center_x < (2 * width / 3):
                direction = "in front of you"
            else:
                direction = "on your right"

            # ------------------------------------------
            # DISTANCE ESTIMATION (Size-Based Approximation)
            # ------------------------------------------

            box_width = x2 - x1
            box_height = y2 - y1
            box_area = box_width * box_height

            # Compare object size to entire frame
            relative_size = box_area / frame_area

            if relative_size > 0.30:
                distance = "very close"
            elif relative_size > 0.05:
                distance = "nearby"
            else:
                distance = "far"


            # ------------------------------------------
            # EMERGENCY WARNING (Safety Override)
            # Triggered when object is very close
            # and directly in front of user
            # ------------------------------------------

            if distance == "very close" and direction == "in front of you":

                # Only trigger once per emergency event
                if not emergency_active:

                    warning_message = f"Warning! {class_name} very close in front of you"

                    threading.Thread(
                        target=speak,
                        args=(warning_message,)
                    ).start()

                    print("EMERGENCY:", warning_message)

                    emergency_active = True

            else:
                # Reset emergency state when condition no longer true
                emergency_active = False


            # Add normal announcement to environment list
            detected_announcements.append(
                f"{class_name} {distance} {direction}"
            )
            
        # --------------------------------------------------
        # UPDATE TRACKER WITH CURRENT FRAME DETECTIONS
        # --------------------------------------------------
        """DeepSORT assigns persistent IDs to detected people
            so that each person can be tracked across frames."""
            
        tracks = tracker.update_tracks(tracker_detections, frame=frame)
        print("Active tracks:", len(tracks)) #temporary debug statement
        
        # --------------------------------------------------
        # FIND THE CLOSEST PERSON
        # --------------------------------------------------
        """We prioritize the closest person because they
            represent the most immediate collision risk."""
            
        closest_person = None
        largest_height = 0

        for track in tracks:

            if not track.is_confirmed():
                continue

            l, t, r, b = track.to_ltrb()

            person_height = b - t

            if person_height > largest_height:
                largest_height = person_height
                closest_person = (l, t, r, b)
                
        # --------------------------------------------------
        # DETERMINE POSITION AND DISTANCE OF CLOSEST PERSON
        # --------------------------------------------------
            
        if closest_person is not None:

            l, t, r, b = closest_person

            center_x = (l + r) / 2

            if center_x < width / 3:
                direction = "on your left"
            elif center_x < (2 * width / 3):
                direction = "in front of you"
            else:
                direction = "on your right"

            person_height = b - t
            relative_size = (person_height * (r - l)) / frame_area

            if relative_size > 0.15:
                distance = "very close"
            elif relative_size > 0.05:
                distance = "nearby"
            else:
                distance = "far"

            detected_announcements.append(
                f"person {distance} {direction}"
            )


    # ==================================================
    # EVENT-BASED SPEECH SYSTEM
    # ==================================================
    # Speaks only when environment changes

    current_set = set(detected_announcements)

    current_time = time.time()

    # Speak only if environment changed AND cooldown passed
    if (
        current_set != last_announcements
        and current_set
        and (current_time - last_speech_time > speech_cooldown)
    ):

        announcement = ", ".join(current_set)

        threading.Thread(
            target=speak,
            args=(announcement,)
        ).start()

        print("Speaking:", announcement)

        # Update memory of last environment state
        last_announcements = current_set
        last_speech_time = current_time
 

    # ----------------------------------------------
    # DISPLAY VIDEO OUTPUT
    # ----------------------------------------------
    cv2.imshow("Assistive Vision - Object Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# ==================================================
# CLEANUP
# ==================================================

cap.release()
cv2.destroyAllWindows()