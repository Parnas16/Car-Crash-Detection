import cv2, os, math, csv, shutil
import numpy as np
from ultralytics import YOLO
from geopy.geocoders import Nominatim

import sys

# Allow video path as argument from Flask
if len(sys.argv) > 1:
    VIDEO_PATH = sys.argv[1]
else:
    VIDEO_PATH = r"C:\Users\HI\Downloads\crash_results (2)finnn\crash_results\Video1.mp4"  # fallback if none given

OUTPUT_DIR = "crash_results"
CLIP_DURATION = 3  # seconds before/after crash
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Video setup
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"🎥 Loaded video: {frame_count} frames @ {fps:.1f} FPS ({width}x{height})")

# Helper: IoU for collision overlap
def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter + 1e-6)

# Prepare annotated output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
annotated_path = os.path.join(OUTPUT_DIR, "annotated_output.mp4")
writer = cv2.VideoWriter(annotated_path, fourcc, fps, (width, height))

# CSV setup
csv_path = os.path.join(OUTPUT_DIR, "crash_timestamps.csv")
csvfile = open(csv_path, mode='w', newline='')
fieldnames = ['Frame', 'Time (s)', 'Status', 'Confidence']
writer_csv = csv.DictWriter(csvfile, fieldnames=fieldnames)
writer_csv.writeheader()

crash_events = []
frame_idx = 0
print("🚦 Starting crash detection...")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    results = model(frame, verbose=False)[0]
    vehicles = []
    vehicle_confs = []

    for box in results.boxes:
        cls = int(box.cls[0])
        if cls in [2, 3, 5, 7]:  # vehicles
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            vehicles.append([x1, y1, x2, y2])
            vehicle_confs.append(conf)

    # Detect collisions
    crash_detected = False
    for i in range(len(vehicles)):
        for j in range(i + 1, len(vehicles)):
            if bbox_iou(vehicles[i], vehicles[j]) > 0.08:
                t = frame_idx / fps
                crash_events.append({"time_s": t, "frame": frame_idx})
                conf_i, conf_j = vehicle_confs[i], vehicle_confs[j]
                writer_csv.writerow({
                    'Frame': frame_idx,
                    'Time (s)': round(t, 2),
                    'Status': 'Crash Detected',
                    'Confidence': f"{conf_i:.2f}, {conf_j:.2f}"
                })
                crash_detected = True

                cv2.putText(frame, f"⚠ CRASH @ {t:.2f}s", (60, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                cv2.putText(frame, f"Conf: {conf_i:.2f}, {conf_j:.2f}", (60, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                break
        if crash_detected:
            break

    # Draw all vehicles with confidence
    for (x1, y1, x2, y2), conf in zip(vehicles, vehicle_confs):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    writer.write(frame)

    # Display the video with detection
    cv2.imshow("Car Crash Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if frame_idx % int(fps * 2) == 0:
        print(f"Processed {frame_idx}/{frame_count} frames...")

cap.release()
writer.release()
csvfile.close()
cv2.destroyAllWindows()

# Filter duplicate crash events
unique_events = []
for e in crash_events:
    if not unique_events or abs(e['frame'] - unique_events[-1]['frame']) > fps * 1.5:
        unique_events.append(e)

# === CLIP GENERATION ===
def save_clip(start_sec, end_sec, clip_name):
    cap2 = cv2.VideoCapture(VIDEO_PATH)
    cap2.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000)
    out_clip_path = os.path.join(OUTPUT_DIR, clip_name)
    writer2 = cv2.VideoWriter(out_clip_path, fourcc, fps, (width, height))
    while True:
        current_time = cap2.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if current_time > end_sec:
            break
        ret, frame = cap2.read()
        if not ret:
            break
        writer2.write(frame)
    cap2.release()
    writer2.release()
    return out_clip_path

print("\n=== 🚨 Crash Detection Report ===")
if not unique_events:
    print("✅ No crash-like events detected.")
else:
    for i, e in enumerate(unique_events, 1):
        start = max(0, e['time_s'] - CLIP_DURATION)
        end = min(frame_count / fps, e['time_s'] + CLIP_DURATION)
        clip = save_clip(start, end, f"crash_clip_{i}.mp4")
        print(f"⚠️ Event {i}: Crash near {e['time_s']:.2f}s — clip saved as {clip}")

print(f"\n🎬 Annotated full video: {annotated_path}")
print(f"📄 Crash timestamps saved to: {csv_path}")

# === SAFETY GUIDE ===
print("\n=== 🏥 Emergency Response Guide ===")
print("1️⃣ Stay calm and check surroundings for fire or fuel leaks.")
print("2️⃣ Alert emergency staff immediately.")
print("3️⃣ If safe, move the driver out and keep them still.")
print("4️⃣ Call ambulance and share exact location.")
print("5️⃣ Provide first aid (airway, bleeding, consciousness).")

# === NEARBY HOSPITAL PLACEHOLDER ===
try:
    geolocator = Nominatim(user_agent="car_crash_app")
    location = geolocator.geocode("Hyderabad, India")
    print(f"\n📍 Example nearest hospitals to {location.address}:")
    print(" - Apollo Hospitals, Jubilee Hills")
    print(" - Yashoda Hospital, Somajiguda")
    print(" - Care Hospitals, Banjara Hills")
except:
    print("⚠️ Unable to fetch hospital data (no internet or geopy issue).")

print("\n✅ Analysis complete! Check crash_results/ for clips, CSV, and annotated video.")
