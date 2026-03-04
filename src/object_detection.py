from ultralytics import YOLO
import cv2

# Load a lightweight YOLOv8 model (good for CPU)
model = YOLO("yolov8n.pt")

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access camera")
    exit()

print("Real-time object detection started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run object detection
    results = model(frame, conf=0.4, stream=True)

    for r in results:
        annotated_frame = r.plot()

    # Show results
    cv2.imshow("Assistive Vision - Object Detection", annotated_frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
