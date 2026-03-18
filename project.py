import cv2, os, math, shutil
import numpy as np
from ultralytics import YOLO
from geopy.geocoders import Nominatim

# ========== CONFIG ==========
VIDEO_PATH = "C:\Users\HI\Downloads\crash_results (2)finnn\crash_results\Video2.mp4"  # ← your .mp4 file
OUTPUT_DIR = "crash_results"
CLIP_DURATION = 3                  # seconds before/after crash
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Video setup
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"🎥 Loaded video: {frame_count} frames @ {fps:.1f} FPS ({width}x{height})")

# Helper: IoU for collision overlap
def bbox_iou(a,b):
    ax1,ay1,ax2,ay2 = a
    bx1,by1,bx2,by2 = b
    inter_x1, inter_y1 = max(ax1,bx1), max(ay1,by1)
    inter_x2, inter_y2 = min(ax2,bx2), min(ay2,by2)
    iw, ih = max(0,inter_x2-inter_x1), max(0,inter_y2-inter_y1)
    inter = iw*ih
    area_a = (ax2-ax1)*(ay2-ay1)
    area_b = (bx2-bx1)*(by2-by1)
    return inter / (area_a + area_b - inter + 1e-6)

# Prepare annotated output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
annotated_path = os.path.join(OUTPUT_DIR, "annotated_output.mp4")
writer = cv2.VideoWriter(annotated_path, fourcc, fps, (width, height))

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

    for box in results.boxes:
        cls = int(box.cls[0])
        if cls in [2, 3, 5, 7]:  # vehicles
            x1,y1,x2,y2 = box.xyxy[0].tolist()
            vehicles.append([int(x1),int(y1),int(x2),int(y2)])

    # Detect collisions
    for i in range(len(vehicles)):
        for j in range(i+1, len(vehicles)):
            if bbox_iou(vehicles[i], vehicles[j]) > 0.08:
                t = frame_idx / fps
                crash_events.append({"time_s": t, "frame": frame_idx})
                cv2.putText(frame, f"CRASH @ {t:.2f}s!", (60,70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

    for (x1,y1,x2,y2) in vehicles:
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

    writer.write(frame)
    if frame_idx % int(fps*2) == 0:
        print(f"Processed {frame_idx}/{frame_count} frames...")

cap.release()
writer.release()

# Filter duplicate events
unique_events = []
for e in crash_events:
    if not unique_events or abs(e['frame'] - unique_events[-1]['frame']) > fps*1.5:
        unique_events.append(e)

# === CLIP GENERATION ===
def save_clip(start_sec, end_sec, clip_name):
    cap2 = cv2.VideoCapture(VIDEO_PATH)
    cap2.set(cv2.CAP_PROP_POS_MSEC, start_sec*1000)
    out_clip_path = os.path.join(OUTPUT_DIR, clip_name)
    writer = cv2.VideoWriter(out_clip_path, fourcc, fps, (width, height))
    while True:
        current_time = cap2.get(cv2.CAP_PROP_POS_MSEC)/1000
        if current_time > end_sec:
            break
        ret, frame = cap2.read()
        if not ret:
            break
        writer.write(frame)
    cap2.release()
    writer.release()
    return out_clip_path

print("\n=== 🚨 Crash Detection Report ===")
if not unique_events:
    print("✅ No crash-like events detected.")
else:
    for i,e in enumerate(unique_events, 1):
        start = max(0, e['time_s'] - CLIP_DURATION)
        end   = min(frame_count/fps, e['time_s'] + CLIP_DURATION)
        clip = save_clip(start, end, f"crash_clip_{i}.mp4")
        print(f"⚠️ Event {i}: Crash near {e['time_s']:.2f}s — clip saved as {clip}")

print(f"\n🎬 Annotated full video: {annotated_path}")

# === SAFETY GUIDE ===
print("\n=== 🏥 Emergency Response Guide ===")
print("1️⃣ Stay calm and check surroundings for fire or fuel leaks.")
print("2️⃣ Alert race control or emergency staff immediately.")
print("3️⃣ If safe, move the driver out and keep them still.")
print("4️⃣ Call ambulance and share exact location.")
print("5️⃣ Provide first aid (airway, bleeding, consciousness).")

# === NEARBY HOSPITAL PLACEHOLDER ===
# (You can add actual coordinates of the racetrack here)
try:
    geolocator = Nominatim(user_agent="f1_crash_app")
    location = geolocator.geocode("Hyderabad, India")
    print(f"\n📍 Example nearest hospitals to {location.address}:")
    print(" - Apollo Hospitals, Jubilee Hills")
    print(" - Yashoda Hospital, Somajiguda")
    print(" - Care Hospitals, Banjara Hills")
except:
    print("⚠️ Unable to fetch hospital data (no internet or geopy issue).")

print("\n✅ Analysis complete! Check crash_results/ for clips and reports.")
