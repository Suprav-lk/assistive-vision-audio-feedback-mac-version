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
last_announcements = ""

# Tracks if emergency warning is currently active
emergency_active = False

# Minimum time between speech announcements
speech_cooldown = 2.5   # seconds

# Timestamp of last speech
last_speech_time = 0

# Emergency warning cooldown
warning_cooldown = 4

# Timestamp of last warning
last_warning_time = 0


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
    # Tracks if an emergency object exists in this frame
    emergency_detected_this_frame = False
    
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
    
    """ Instead of announcing all objects, we track only
        the closest obstacle in the frame for navigation safety"""
    # Track the closest obstacle in the frame
    closest_object = None
    closest_size = 0

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
            # FIND CLOSEST OBJECT
            # ------------------------------------------
            # Larger bounding box area means object is closer

            if relative_size > closest_size:

                closest_size = relative_size

                closest_object = {
                    "name": class_name,
                    "distance": distance,
                    "direction": direction
                }

           # ------------------------------------------
            # EMERGENCY WARNING (Safety Override)
            # Triggered when object is very close
            # and directly in front of user
            # ------------------------------------------

            if distance == "very close" and direction == "in front of you":

                # Mark that an emergency exists in this frame
                emergency_detected_this_frame = True

                # Trigger warning only once
                if not emergency_active:

                    warning_message = f"Warning. {class_name} very close in front of you"

                    threading.Thread(
                        target=speak,
                        args=(warning_message,)
                    ).start()

                    print("EMERGENCY:", warning_message)

                    emergency_active = True
            
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


    # ==================================================
    # SPEAK ONLY THE CLOSEST OBSTACLE
    # ==================================================

    current_time = time.time()

    if closest_object:

        announcement = (
            f"{closest_object['name']} "
            f"{closest_object['distance']} "
            f"{closest_object['direction']}"
        )

        # Speak only if announcement changed and cooldown passed
        if (
            announcement != last_announcements
            and (current_time - last_speech_time > speech_cooldown)
        ):

            threading.Thread(
                target=speak,
                args=(announcement,)
            ).start()

            print("Speaking:", announcement)

            last_announcements = announcement
            last_speech_time = current_time
    
    # Reset emergency state if nothing dangerous was detected
    if not emergency_detected_this_frame:
        emergency_active = False
        
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