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
    "table",
    "chair",
    "sofa",
    "bed",
    "door",
    "staircase",
    "dining table",
    "wall",
    "window",
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


# ----------------------------------------------
# Stability filter for closest obstacle
# Object must remain stable before announcement
# ----------------------------------------------

stable_object = None          # currently observed closest obstacle
stable_start_time = 0         # when it first appeared
stability_threshold = 2.0     # seconds required before speaking

# Emergency warning cooldown
warning_cooldown = 4

# Timestamp of last warning
last_warning_time = 0


speech_lock = threading.Lock()


# ==================================================
# APPROACHING OBJECT TRACKER (NEW FEATURE)
# ==================================================

approach_tracker = {}

APPROACH_THRESHOLD = 1.02    # 2% increase in size
APPROACH_COUNT_REQUIRED = 3 # frames needed to confirm
APPROACH_COOLDOWN = 5        # seconds between alerts


# ==================================================
# 5. SPEECH FUNCTION (Non-blocking)
# ==================================================
# Runs speech in a separate thread so video doesn't freeze

def speak(text):
    """
    Ensures only ONE speech runs at a time.
    Prevents backlog and delayed audio.
    """
    if speech_lock.locked():
        return  # Skip if already speaking (drop old messages)

    def run():
        with speech_lock:
            os.system(f"say {text}")

    threading.Thread(target=run).start()


def check_approaching(object_id, area):
    """
    Detects if an object is approaching based on bounding box growth.
    """

    current_time = time.time()

    if object_id not in approach_tracker:
        approach_tracker[object_id] = {
            "areas": [],
            "increase_count": 0,
            "last_alert": 0
        }

    data = approach_tracker[object_id]

    # Store recent area values (last 5 frames)
    data["areas"].append(area)
    if len(data["areas"]) > 5:
        data["areas"].pop(0)

    # Check if size is increasing
    if len(data["areas"]) >= 2:
        if data["areas"][-1] > data["areas"][-2] * APPROACH_THRESHOLD:
            data["increase_count"] += 1
            print(f"{object_id} increasing | count: {data['increase_count']}")
        else:
            data["increase_count"] = 0

    # Confirm approaching
    if data["increase_count"] >= APPROACH_COUNT_REQUIRED:
        if current_time - data["last_alert"] > APPROACH_COOLDOWN:
            data["last_alert"] = current_time
            data["increase_count"] = 0
            return True

    return False


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
    if frame_count % 2 != 0:
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
    largest_relative_size = 0

    # Get frame size (used for distance estimation)
    height, width, _ = frame.shape
    frame_area = width * height


    # ==================================================
    # PROCESS EACH DETECTED OBJECT
    # ==================================================
    annotated_frame = frame.copy()
    for r in results:
        annotated_frame = frame.copy()
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
            
            # DEBUG: Print object size
            print(f"{class_name} area: {box_area}") #-------------------------------------------jdflaksjflakjfsals-------------
            
            # ------------------------------------------
            # APPROACHING OBJECT DETECTION 
            # ------------------------------------------

            # Create simple object ID (based on position + class)
            object_id = f"{class_name}_{int(x1/100)}_{int(y1/100)}"

            # check if object is approaching and cooldown has passed
            is_approaching = check_approaching(object_id, box_area)

            if is_approaching and (time.time() - last_speech_time > speech_cooldown):

                approaching_message = f"{class_name} approaching {direction}"

                speak(approaching_message)

                print("APPROACHING:", approaching_message)

                last_speech_time = time.time()

                # Mark emergency as already handled to prevent override
                emergency_active = True

                continue
                                
            # ------------------------------------------  

            # Compare object size to entire frame
            relative_size = box_area / frame_area

            if relative_size > 0.45:
                distance = "very close"
            elif relative_size > 0.05:
                distance = "nearby"
            else:
                distance = "far"

            # ------------------------------------------
            # FIND CLOSEST OBJECT
            # ------------------------------------------
            # Larger bounding boxes indicate objects closer to the camera.
            # We keep only the largest object so that the system announces
            # the closest obstacle instead of overwhelming the user.
            
            if relative_size > largest_relative_size:

                largest_relative_size = relative_size

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

                    speak(warning_message)

                    print("EMERGENCY:", warning_message)

                    emergency_active = True
            
        # --------------------------------------------------
        # UPDATE TRACKER WITH CURRENT FRAME DETECTIONS
        # --------------------------------------------------
        """DeepSORT assigns persistent IDs to detected people
            so that each person can be tracked across frames."""
            
        tracks = tracker.update_tracks(tracker_detections, frame=frame)
        print("Active tracks:", len(tracks)) #temporary debug statement
        
        
    # ==================================================
    # SPEAK ONLY THE CLOSEST OBSTACLE
    # ==================================================

    if closest_object is None:
        stable_object = None
    
    current_time = time.time()

    if closest_object:

        announcement = (
            f"{closest_object['name']} "
            f"{closest_object['distance']} "
            f"{closest_object['direction']}"
        )

        if announcement != stable_object:
            # New object detected → start stability timer
            stable_object = announcement
            stable_start_time = current_time

        else:
            # Same object still detected
            stable_duration = current_time - stable_start_time

            if (
                stable_duration > stability_threshold
                and announcement != last_announcements
                and (current_time - last_speech_time > speech_cooldown)
            ):

                speak(announcement)

                print("Speaking:", announcement)

                last_announcements = announcement
                last_speech_time = current_time
    
    # Reset emergency state if nothing dangerous was detected
    if not emergency_detected_this_frame:
        emergency_active = False
        
    # Reset last announcement if nothing detected
    if not closest_object:
        last_announcements = ""
        
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