import threading
import queue
import json
import time
import os

import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# === Vosk offline speech recognition ===
import vosk
import pyaudio
# === END ===

# ==Emergency alerting (Twilio)===
import json
import requests
from twilio.rest import Client
# === END ===


# ==================================================
# 1. LOAD YOLO MODEL
# ==================================================

model = YOLO("yolov8n.pt")
model.to("mps")


# ==================================================
# PERSON TRACKER
# ==================================================

tracker = DeepSort(
    max_age=30,
    n_init=3,
    max_cosine_distance=0.4
)



# ==================================================
# 3. PRIORITY OBJECTS
# ==================================================

important_objects = [
    "person", "table", "chair", "sofa", "bed",
    "door", "staircase", "dining table", "wall", "window",
]

# ==================================================
# 4. SYSTEM STATE VARIABLES
# ==================================================

frame_count = 0
last_announcements = ""
emergency_active = False
speech_cooldown = 2.5
last_speech_time = 0
stable_object = None
stable_start_time = 0
stability_threshold = 2.0
warning_cooldown = 4
last_warning_time = 0
speech_lock = threading.Lock()

approach_tracker = {}
APPROACH_THRESHOLD = 1.02
APPROACH_COUNT_REQUIRED = 3
APPROACH_COOLDOWN = 5

# === NEW: System running flag — controlled by voice commands ===
system_running = threading.Event()   # Acts as an ON/OFF switch
system_running.set()                 # Start with system ON by default
# === END NEW ===



# ==================================================
# 5. SPEECH FUNCTION (Non-blocking)
# ==================================================

def speak(text):
    if speech_lock.locked():
        return
    def run():
        with speech_lock:
            os.system(f"say {text}")
    threading.Thread(target=run).start()


# ==================================================
# === EMERGENCY CALL SYSTEM ===
# ==================================================

def load_config(config_path="config.json"):
    """Loads Twilio credentials from config.json."""
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("[Emergency] ERROR: config.json not found.")
        return None
    except json.JSONDecodeError:
        print("[Emergency] ERROR: config.json is invalid.")
        return None


def load_contacts(contacts_path="contacts.json"):
    """Loads emergency contacts from contacts.json."""
    try:
        with open(contacts_path, "r") as f:
            data = json.load(f)
            return data.get("contacts", [])
    except FileNotFoundError:
        print("[Emergency] ERROR: contacts.json not found.")
        return []
    except json.JSONDecodeError:
        print("[Emergency] ERROR: contacts.json is invalid.")
        return []


def get_location():
    """
    Gets approximate location via IP address.
    Returns a string like 'Kathmandu, Nepal (27.7172, 85.3240)'
    Falls back to 'Location unavailable' if no internet.
    """
    try:
        response = requests.get("https://ipapi.co/json/", timeout=5)
        data = response.json()
        city = data.get("city", "Unknown city")
        country = data.get("country_name", "Unknown country")
        lat = data.get("latitude", "?")
        lon = data.get("longitude", "?")
        return f"{city}, {country} (Lat: {lat}, Lon: {lon})"
    except Exception as e:
        print(f"[Emergency] Could not get location: {e}")
        return "Location unavailable"


def find_contact(name, contacts):
    """
    Finds a contact by name (case-insensitive).
    Returns the contact dict or None if not found.
    """
    name = name.lower().strip()
    for contact in contacts:
        if contact["name"].lower() == name:
            return contact
    return None


def make_emergency_call(contact, config, location):
    """
    Makes a phone call and sends an SMS to the contact via Twilio.
    """
    try:
        client = Client(
            config["twilio_account_sid"],
            config["twilio_auth_token"]
        )

        from_number = config["twilio_phone_number"]
        to_number = contact["phone"]
        contact_name = contact["name"].capitalize()

        # --- Send SMS with location ---
        sms_message = (
            f"EMERGENCY! The visually impaired user needs help. "
            f"Location: {location}"
        )

        client.messages.create(
            body=sms_message,
            from_=from_number,
            to=to_number
        )

        print(f"[Emergency] SMS sent to {contact_name} at {to_number}")

        # --- Make phone call ---
        # Twilio reads this message aloud when the contact picks up
        call_message = (
            f"<Response><Say>Emergency. The visually impaired user needs help. "
            f"Please respond immediately. This call was made automatically "
            f"by their assistive vision system.</Say></Response>"
        )

        call = client.calls.create(
            twiml=call_message,
            from_=from_number,
            to=to_number
        )

        print(f"[Emergency] Call initiated to {contact_name}. Call SID: {call.sid}")
        return True

    except Exception as e:
        print(f"[Emergency] ERROR making call/SMS: {e}")
        return False


def handle_emergency_call(name):
    """
    Main handler triggered by voice command.
    Loads config, finds contact, gets location, makes call + SMS.
    Runs in a separate thread so it doesn't block video loop.
    """
    def run():
        print(f"[Emergency] Handling emergency call for: {name}")
        speak(f"Calling {name}. Please wait.")

        config = load_config()
        if not config:
            speak("Emergency call failed. Configuration not found.")
            return

        contacts = load_contacts()
        if not contacts:
            speak("Emergency call failed. No contacts found.")
            return

        contact = find_contact(name, contacts)
        if not contact:
            speak(f"Contact {name} not found.")
            print(f"[Emergency] Contact '{name}' not in contacts.json")
            return

        location = get_location()
        print(f"[Emergency] Location: {location}")

        success = make_emergency_call(contact, config, location)

        if success:
            speak(f"Emergency call and message sent to {name}.")
        else:
            speak(f"Emergency call to {name} failed. Please try again.")

    threading.Thread(target=run, daemon=True).start()

# === END EMERGENCY CALL SYSTEM ===

# ==================================================
# APPROACHING OBJECT TRACKER
# ==================================================

def check_approaching(object_id, area):
    current_time = time.time()
    if object_id not in approach_tracker:
        approach_tracker[object_id] = {
            "areas": [], "increase_count": 0, "last_alert": 0
        }
    data = approach_tracker[object_id]
    data["areas"].append(area)
    if len(data["areas"]) > 5:
        data["areas"].pop(0)
    if len(data["areas"]) >= 2:
        if data["areas"][-1] > data["areas"][-2] * APPROACH_THRESHOLD:
            data["increase_count"] += 1
        else:
            data["increase_count"] = 0
    if data["increase_count"] >= APPROACH_COUNT_REQUIRED:
        if current_time - data["last_alert"] > APPROACH_COOLDOWN:
            data["last_alert"] = current_time
            data["increase_count"] = 0
            return True
    return False


# ==================================================
# === NEW: VOICE COMMAND LISTENER (runs in background) ===
# ==================================================

def load_vosk_model(model_path="vosk-model-small-en-us"):
    """
    Loads the Vosk offline speech recognition model.
    Returns None if the model folder is not found.
    """
    if not os.path.exists(model_path):
        print(
            f"[Voice] ERROR: Vosk model not found at '{model_path}'.\n"
            "Download from https://alphacephei.com/vosk/models and unzip here."
        )
        return None
    try:
        vosk.SetLogLevel(-1)   # Suppress Vosk internal logs
        return vosk.Model(model_path)
    except Exception as e:
        print(f"[Voice] ERROR loading Vosk model: {e}")
        return None


def voice_command_listener(vosk_model, system_running_event):
    """
    Runs continuously in a background thread.
    Listens for:
      - "start system"  → sets system_running event (enables detection)
      - "stop system"   → clears system_running event (pauses detection)

    Args:
        vosk_model: loaded Vosk Model object
        system_running_event: threading.Event used as ON/OFF flag
    """
    SAMPLE_RATE = 16000
    CHUNK = 4000              # ~250ms of audio per chunk — low latency

    # --- Open microphone stream ---
    try:
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        stream.start_stream()
    except Exception as e:
        print(f"[Voice] ERROR: Cannot open microphone — {e}")
        print("[Voice] Voice commands disabled. Check microphone permissions.")
        return

    recognizer = vosk.KaldiRecognizer(vosk_model, SAMPLE_RATE)
    print("[Voice] Listener active. Say 'Start system' or 'Stop system'.")

    while True:
        try:
            audio_data = stream.read(CHUNK, exception_on_overflow=False)
        except OSError as e:
            print(f"[Voice] Microphone read error: {e}. Retrying...")
            time.sleep(1)
            continue

        # Feed audio chunk to Vosk recognizer
        if recognizer.AcceptWaveform(audio_data):
            result = json.loads(recognizer.Result())
            text = result.get("text", "").lower().strip()

            if not text:
                continue

            print(f"[Voice] Heard: '{text}'")

            # --- Command matching ---
            if "start system" in text:
                if not system_running_event.is_set():
                    print("[Voice] Command: START SYSTEM")
                    system_running_event.set()
                    speak("System started")
                else:
                    print("[Voice] System already running.")

            elif "stop system" in text:
                if system_running_event.is_set():
                    print("[Voice] Command: STOP SYSTEM")
                    system_running_event.clear()
                    speak("System stopped")
                else:
                    print("[Voice] System already stopped.")
                    
                    
            # === Emergency call command ===
            elif "emergency call" in text:
                # Extract name after "emergency call"
                # e.g. "emergency call mom" → name = "mom"
                parts = text.split("emergency call", 1)
                if len(parts) > 1:
                    contact_name = parts[1].strip()
                    if contact_name:
                        print(f"[Voice] Command: EMERGENCY CALL → {contact_name}")
                        handle_emergency_call(contact_name)
                    else:
                        speak("Please say a name after emergency call.")
                else:
                    speak("Please say a name after emergency call.")
            # === END EMERGENCY CALL SYSTEM ===

    # Cleanup (reached only if loop is broken externally)
    stream.stop_stream()
    stream.close()
    audio.terminate()




# ==================================================
# === NEW: CAMERA INITIALIZER / RELEASER HELPERS ===
# ==================================================

def open_camera():
    """Opens camera and returns capture object, or None on failure."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Camera] ERROR: Cannot access camera.")
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    print("[Camera] Camera opened.")
    return cap


def release_camera(cap):
    """Safely releases the camera."""
    if cap is not None and cap.isOpened():
        cap.release()
        print("[Camera] Camera released.")

# === END NEW ===


# ==================================================
# 6. MAIN ENTRY POINT
# ==================================================

def main():
    global frame_count, last_announcements, emergency_active
    global last_speech_time, stable_object, stable_start_time, last_warning_time

    # --- Load Vosk model (once at startup) ---
    # === NEW ===
    vosk_model = load_vosk_model("vosk-model-small-en-us")

    if vosk_model:
        # Launch voice listener in a daemon thread
        # Daemon = auto-killed when main program exits
        voice_thread = threading.Thread(
            target=voice_command_listener,
            args=(vosk_model, system_running),
            daemon=True
        )
        voice_thread.start()
    else:
        print("[Voice] Running without voice commands.")
    # === END NEW ===

    print("Assistive Vision started. Press 'q' to quit.")
    speak("Assistive Vision ready. Say start system to begin.")

    cap = None   # Camera starts as None; opened when system turns ON

    # ==================================================
    # MAIN LOOP
    # ==================================================
    while True:

        # === NEW: Handle system ON/OFF state ===
        if not system_running.is_set():
            # System is OFF — release camera if it's open, wait
            release_camera(cap)
            cap = None
            frame_count = 0
            last_announcements = ""
            stable_object = None

            # Wait efficiently until system is turned back ON
            # Check every 0.3s so the loop stays responsive to 'q' key
            cv2.waitKey(300)
            continue

        # System is ON — open camera if not already open
        if cap is None:
            cap = open_camera()
            if cap is None:
                time.sleep(1)
                continue
        # === END NEW ===

        ret, frame = cap.read()
        emergency_detected_this_frame = False

        if not ret:
            print("[Camera] Frame read failed. Retrying...")
            time.sleep(0.1)
            continue

        frame_count += 1

        # Skip alternate frames for performance
        if frame_count % 2 != 0:
            cv2.imshow("Assistive Vision - Object Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # ----------------------------------------------
        # RUN OBJECT DETECTION
        # ----------------------------------------------
        results = model(frame, conf=0.3, stream=True)

        closest_object = None
        largest_relative_size = 0

        height, width, _ = frame.shape
        frame_area = width * height

        annotated_frame = frame.copy()

        for r in results:
            annotated_frame = r.plot()
            tracker_detections = []

            for box in r.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]

                if class_name == "person":
                    x1, y1, x2, y2 = box.xyxy[0]
                    w, h = x2 - x1, y2 - y1
                    confidence = float(box.conf[0])
                    tracker_detections.append(
                        ([int(x1), int(y1), int(w), int(h)], confidence, "person")
                    )

                if class_name not in important_objects:
                    continue

                x1, y1, x2, y2 = box.xyxy[0]
                center_x = (x1 + x2) / 2

                if center_x < width / 3:
                    direction = "on your left"
                elif center_x < (2 * width / 3):
                    direction = "in front of you"
                else:
                    direction = "on your right"

                box_width = x2 - x1
                box_height = y2 - y1
                box_area = box_width * box_height

                print(f"{class_name} area: {box_area}")

                object_id = f"{class_name}_{int(x1/100)}_{int(y1/100)}"
                is_approaching = check_approaching(object_id, box_area)

                if is_approaching and (time.time() - last_speech_time > speech_cooldown):
                    approaching_message = f"{class_name} approaching {direction}"
                    speak(approaching_message)
                    print("APPROACHING:", approaching_message)
                    last_speech_time = time.time()
                    emergency_active = True
                    continue

                relative_size = box_area / frame_area

                if relative_size > 0.45:
                    distance = "very close"
                elif relative_size > 0.05:
                    distance = "nearby"
                else:
                    distance = "far"

                if relative_size > largest_relative_size:
                    largest_relative_size = relative_size
                    closest_object = {
                        "name": class_name,
                        "distance": distance,
                        "direction": direction
                    }

                if distance == "very close" and direction == "in front of you":
                    emergency_detected_this_frame = True
                    if not emergency_active:
                        warning_message = f"Warning. {class_name} very close in front of you"
                        speak(warning_message)
                        print("EMERGENCY:", warning_message)
                        emergency_active = True

            tracks = tracker.update_tracks(tracker_detections, frame=frame)
            print("Active tracks:", len(tracks))

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
                stable_object = announcement
                stable_start_time = current_time
            else:
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

        if not emergency_detected_this_frame:
            emergency_active = False

        if not closest_object:
            last_announcements = ""

        cv2.imshow("Assistive Vision - Object Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # ==================================================
    # CLEANUP
    # ==================================================
    release_camera(cap)
    cv2.destroyAllWindows()
    print("Assistive Vision shut down.")


if __name__ == "__main__":
    main()