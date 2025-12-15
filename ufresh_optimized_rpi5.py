import cv2
import threading
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from ultralytics import YOLO
import csv
import os
from datetime import datetime, date
import pandas as pd
import time
import queue
from collections import deque
import numpy as np

# ----------------- YOLO + ByteTrack Setup -----------------
model = YOLO("best.pt")  # change to your model path
video_path = "output_2025-04-01_16-56-10.mp4"

# Performance optimizations for Raspberry Pi 5
FRAME_SKIP = 0  # Process EVERY frame for accuracy
DETECTION_CONFIDENCE = 0.25  # Lower confidence for more detections
TRACKER_CONFIDENCE = 0.3
MAX_TRACK_AGE = 50  # Keep tracks longer
BUFFER_SIZE = 10  # Frame buffer size for smooth processing

# Your categories
CATEGORIES = [
    'EMPTY_CONVEYER_BELT',
    'NO_DETECTION_POSSIBLE',
    'UFRESH_BUTTERMILK_1',
    'UFRESH_CURD_POUCH_1',
    'UFRESH_CURD_POUCH_2',
    'UFRESH_DAHI_SPECIAL_MILK_1',
    'UFRESH_DAHI_SPECIAL_MILK_2',
    'UFRESH_DAHI_SPECIAL_MILK_3',
    'UFRESH_GOLD_MILK_1',
    'UFRESH_GOLD_MILK_2',
    'UFRESH_KHATI_CHAAS_1',
    'UFRESH_MASALA_BUTTERMILK_1',
    'UFRESH_TAZA_MILK_1',
    'UFRESH_TEA_SPECIAL_MILK_1'
]

line_y = 320
counted_ids = set()
previous_y = {}
track_history = {}  # Store track history for better detection

# Enhanced tracking for high-speed detection
class TrackingState:
    def __init__(self):
        self.position_history = deque(maxlen=10)  # Keep last 10 positions
        self.last_seen = 0
        self.crossed_line = False
        
track_states = {}

# Dict for per-class counts
counts_per_class = {cat: 0 for cat in CATEGORIES}

# Daily CSV file
CSV_FILE = "daily_counts.csv"
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "Category", "Count"])

# Track current date for resetting daily counts
current_date = date.today()

# Frame processing queue for better performance
frame_queue = queue.Queue(maxsize=BUFFER_SIZE)
result_queue = queue.Queue(maxsize=BUFFER_SIZE)

# Performance monitoring
fps_counter = 0
fps_start_time = time.time()
current_fps = 0

# ----------------- Tkinter GUI Setup -----------------
root = tk.Tk()
root.title("Optimized Crate Counter - Raspberry Pi 5")

# Detect Raspberry Pi screen resolution
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.geometry(f"{screen_width}x{screen_height}+0+0")

notebook = ttk.Notebook(root)
notebook.pack(fill="both", expand=True)

# --- Tab 1: Live Tracking ---
tab1 = ttk.Frame(notebook)
notebook.add(tab1, text="Live Tracking")

video_label = tk.Label(tab1)
video_label.pack(fill="both", expand=True)

# Performance info
perf_frame = tk.Frame(tab1)
perf_frame.pack(fill="x", pady=5)

fps_label = tk.Label(
    perf_frame, 
    text="FPS: 0", 
    font=("Arial", 12), 
    fg="orange"
)
fps_label.pack(side="left", padx=10)

live_total_label = tk.Label(
    perf_frame, 
    text="Live Total: 0", 
    font=("Arial", max(14, screen_height // 40)), 
    fg="green"
)
live_total_label.pack(side="right", padx=10)

# --- Tab 2: Category Counts ---
tab2 = ttk.Frame(notebook)
notebook.add(tab2, text="Category Counts")

total_label = tk.Label(
    tab2, 
    text="Total Count: 0", 
    font=("Arial", max(16, screen_height // 35)), 
    fg="blue"
)
total_label.pack(pady=10)

tree = ttk.Treeview(tab2, columns=("Category", "Count"), show='headings')
tree.heading("Category", text="Category")
tree.heading("Count", text="Count")
tree.pack(fill="both", expand=True)

# --- Tab 3: Daily Summary ---
frame3 = ttk.Frame(notebook)
notebook.add(frame3, text="Daily Summary")

date_label = ttk.Label(frame3, text="Select Date:", font=("Arial", max(12, screen_height // 45)))
date_label.pack(pady=5)

date_combo = ttk.Combobox(frame3, state="readonly", font=("Arial", max(11, screen_height // 50)))
date_combo.pack(pady=5)

summary_tree = ttk.Treeview(frame3, columns=("Category", "Count"), show="headings")
summary_tree.heading("Category", text="Category")
summary_tree.heading("Count", text="Count")
summary_tree.pack(fill="both", expand=True, padx=10, pady=10)

# ----------------- Helper Functions -----------------
def reset_daily_counts():
    """Reset counts when the day changes"""
    global counts_per_class, counted_ids, previous_y, track_states
    counts_per_class = {cat: 0 for cat in CATEGORIES}
    counted_ids.clear()
    previous_y.clear()
    track_states.clear()
    live_total_label.config(text="Live Total: 0")
    total_label.config(text="Total Count: 0")
    tree.delete(*tree.get_children())
    for cat in CATEGORIES:
        tree.insert('', 'end', values=(cat, 0))

def update_date_options():
    """Update dropdown with unique dates from CSV"""
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        if "Date" in df.columns:
            unique_dates = sorted(df["Date"].astype(str).unique())
            date_combo["values"] = unique_dates

def show_summary_for_date(event=None):
    """Show category wise summary for selected date"""
    selected_date = date_combo.get()
    if not selected_date:
        return
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        if "Date" in df.columns:
            day_data = df[df["Date"] == selected_date]
            summary_tree.delete(*summary_tree.get_children())
            summary = day_data.groupby("Category")["Count"].sum().reset_index()
            for _, row in summary.iterrows():
                summary_tree.insert("", "end", values=(row["Category"], row["Count"]))

date_combo.bind("<<ComboboxSelected>>", show_summary_for_date)
update_date_options()

def save_daily_count(today, category, count):
    """Save or update per-category daily count in CSV"""
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        # Check if the CSV is empty or doesn't have the required columns
        if df.empty or "Date" not in df.columns:
            df = pd.DataFrame(columns=["Date", "Category", "Count"])
    else:
        df = pd.DataFrame(columns=["Date", "Category", "Count"])
    
    mask = (df["Date"] == today) & (df["Category"] == category)
    if mask.any():
        df.loc[mask, "Count"] = count
    else:
        df = pd.concat([df, pd.DataFrame([[today, category, count]],
                                         columns=["Date", "Category", "Count"])],
                       ignore_index=True)
    df.to_csv(CSV_FILE, index=False)

def enhanced_line_crossing_detection(obj_id, cx, cy, class_name, frame_count):
    """Enhanced line crossing detection with robust interpolation for high-speed objects"""
    if obj_id not in track_states:
        track_states[obj_id] = TrackingState()
    
    track_state = track_states[obj_id]
    track_state.position_history.append((cx, cy, frame_count))
    track_state.last_seen = frame_count
    
    # Keep only last 8 positions for better interpolation
    if len(track_state.position_history) > 8:
        track_state.position_history.popleft()
    
    # Method 1: Simple line crossing check (most reliable)
    if obj_id in previous_y:
        prev_y = previous_y[obj_id]
        if prev_y > line_y and cy <= line_y and obj_id not in counted_ids:
            print(f"✓ COUNTED (Simple): {class_name} (ID:{obj_id}) - Y:{prev_y}→{cy}")
            track_state.crossed_line = True
            return True
    
    # Method 2: Enhanced interpolation for fast objects
    if len(track_state.position_history) >= 3:
        # Check multiple recent positions for line crossing
        for i in range(len(track_state.position_history) - 2):
            p1_x, p1_y, p1_frame = track_state.position_history[i]
            p2_x, p2_y, p2_frame = track_state.position_history[i + 1]
            p3_x, p3_y, p3_frame = track_state.position_history[i + 2]
            
            # Check if line was crossed between any two consecutive points
            if ((p1_y > line_y and p2_y <= line_y) or (p2_y > line_y and p3_y <= line_y)) and obj_id not in counted_ids:
                # Interpolate to find exact crossing point
                if p1_y != p2_y and p1_y > line_y and p2_y <= line_y:
                    # Calculate exact crossing point
                    cross_ratio = (line_y - p1_y) / (p2_y - p1_y)
                    cross_x = p1_x + cross_ratio * (p2_x - p1_x)
                    print(f"✓ COUNTED (Interp): {class_name} (ID:{obj_id}) - Cross at X:{cross_x:.1f}")
                    track_state.crossed_line = True
                    return True
                elif p2_y != p3_y and p2_y > line_y and p3_y <= line_y:
                    cross_ratio = (line_y - p2_y) / (p3_y - p2_y)
                    cross_x = p2_x + cross_ratio * (p3_x - p2_x)
                    print(f"✓ COUNTED (Interp2): {class_name} (ID:{obj_id}) - Cross at X:{cross_x:.1f}")
                    track_state.crossed_line = True
                    return True
    
    # Method 3: Check if object is currently crossing the line (failsafe)
    if cy >= line_y - 5 and cy <= line_y + 5 and obj_id not in counted_ids:
        # Check if object was above the line recently (convert deque to list for slicing)
        recent_positions = list(track_state.position_history)[:-1]
        recent_above = any(pos[1] > line_y for pos in recent_positions)
        if recent_above:
            print(f"✓ COUNTED (Failsafe): {class_name} (ID:{obj_id}) - Near line Y:{cy}")
            track_state.crossed_line = True
            return True
    
    return False

def cleanup_old_tracks(current_frame):
    """Remove old tracks to prevent memory buildup"""
    to_remove = []
    for obj_id, track_state in track_states.items():
        if current_frame - track_state.last_seen > MAX_TRACK_AGE:
            to_remove.append(obj_id)
    
    for obj_id in to_remove:
        if obj_id in track_states:
            del track_states[obj_id]
        if obj_id in previous_y:
            del previous_y[obj_id]

# ----------------- Frame Capture Thread -----------------
def capture_frames():
    """Dedicated thread for frame capture"""
    cap = cv2.VideoCapture(video_path)
    
    # Optimize camera settings for Raspberry Pi 5
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize delay
    if video_path != 0:  # Only set FPS for video files, not camera
        cap.set(cv2.CAP_PROP_FPS, 30)  # Set desired FPS
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video ended or camera disconnected")
            break  # Exit the loop when video ends
            
        frame_count += 1
        
        # Skip frames if needed for performance
        if frame_count % (FRAME_SKIP + 1) != 0:
            continue
            
        # Add frame to queue (non-blocking)
        try:
            frame_queue.put((frame, frame_count), block=False)
        except queue.Full:
            # If queue is full, remove oldest frame
            try:
                frame_queue.get_nowait()
                frame_queue.put((frame, frame_count), block=False)
            except queue.Empty:
                pass
    
    cap.release()
    print("Frame capture thread ended")

# ----------------- Detection Processing Thread -----------------
def process_detections():
    """Dedicated thread for YOLO detection processing"""
    global current_date
    
    while True:
        try:
            frame, frame_count = frame_queue.get(timeout=5.0)
        except queue.Empty:
            print("No frames received for 5 seconds, checking if video ended...")
            continue
            
        today = date.today()
        if today != current_date:
            current_date = today
            reset_daily_counts()

        # Run YOLO detection with optimized parameters
        results = model.track(
            frame, 
            persist=True, 
            tracker="bytetrack.yaml",
            conf=DETECTION_CONFIDENCE,
            iou=0.4,  # Lower IOU for more overlapping detections
            verbose=False  # Reduce console output for performance
        )

        detections = []
        detection_count = 0
        
        if results[0].boxes.id is not None:
            for box in results[0].boxes:
                class_id = int(box.cls[0].item())
                obj_id = int(box.id.item())

                if 0 <= class_id < len(CATEGORIES):
                    class_name = CATEGORIES[class_id]
                else:
                    continue

                if class_name in ["EMPTY_CONVEYER_BELT"]:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                confidence = float(box.conf[0].item())
                
                detection_count += 1
                if detection_count <= 5:  # Print first few detections for debugging
                    print(f"Detection: ID:{obj_id}, Class:{class_name}, Center:({cx},{cy}), Conf:{confidence:.2f}, Line Y:{line_y}")

                # Enhanced line crossing detection
                if enhanced_line_crossing_detection(obj_id, cx, cy, class_name, frame_count):
                    counts_per_class[class_name] += 1
                    counted_ids.add(obj_id)
                    print(f"COUNT UPDATED! {class_name}: {counts_per_class[class_name]}")

                    today_str = today.strftime("%Y-%m-%d")
                    save_daily_count(today_str, class_name, counts_per_class[class_name])
                    update_date_options()
                
                # Update previous_y for next frame comparison
                previous_y[obj_id] = cy

                detections.append({
                    'obj_id': obj_id,
                    'class_name': class_name,
                    'bbox': (x1, y1, x2, y2),
                    'center': (cx, cy),
                    'confidence': confidence
                })
        
        if frame_count % 30 == 0:  # Print status every 30 frames
            total_count = sum(counts_per_class.values())
            print(f"Frame {frame_count}: {detection_count} detections, Total count: {total_count}")

        # Cleanup old tracks periodically
        if frame_count % 30 == 0:  # Every 30 frames
            cleanup_old_tracks(frame_count)

        # Add processed result to result queue
        try:
            result_queue.put((frame, detections, frame_count), block=False)
        except queue.Full:
            # If queue is full, remove oldest result
            try:
                result_queue.get_nowait()
                result_queue.put((frame, detections, frame_count), block=False)
            except queue.Empty:
                pass

# ----------------- Display Thread -----------------
def update_display():
    """Update the GUI display"""
    global fps_counter, fps_start_time, current_fps
    
    try:
        frame, detections, frame_count = result_queue.get_nowait()
        
        # Draw detections on frame
        for detection in detections:
            obj_id = detection['obj_id']
            class_name = detection['class_name']
            x1, y1, x2, y2 = detection['bbox']
            cx, cy = detection['center']
            confidence = detection['confidence']
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} ID:{obj_id} ({confidence:.2f})", 
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Draw center point
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        # Draw counting line (more visible)
        cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 0, 255), 4)  # Thicker red line
        cv2.putText(frame, f"COUNTING LINE Y={line_y}", (30, line_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Draw total count
        total_count = sum(counts_per_class.values())
        cv2.putText(frame, f'Total Count: {total_count}', (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        
        # Calculate and display FPS
        fps_counter += 1
        current_time = time.time()
        if current_time - fps_start_time >= 1.0:
            current_fps = fps_counter / (current_time - fps_start_time)
            fps_counter = 0
            fps_start_time = current_time
            
        cv2.putText(frame, f'FPS: {current_fps:.1f}', (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        # Resize video dynamically to fit screen
        frame = cv2.resize(frame, (screen_width, screen_height // 2))
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)

        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

        # Update GUI elements
        live_total_label.config(text=f"Live Total: {total_count}")
        total_label.config(text=f"Total Count: {total_count}")
        fps_label.config(text=f"FPS: {current_fps:.1f}")

        tree.delete(*tree.get_children())
        for cat, count in counts_per_class.items():
            tree.insert('', 'end', values=(cat, count))
            
    except queue.Empty:
        pass
    
    # Schedule next update
    root.after(33, update_display)  # ~30 FPS GUI updates

# ----------------- Simple Single Thread Processing -----------------
def run_simple_detection():
    """Simple single-threaded detection processing"""
    global current_date, fps_counter, fps_start_time, current_fps
    
    print("=== HIGH-ACCURACY CRATE COUNTER SETTINGS ===")
    print(f"Frame Skip: {FRAME_SKIP} (0 = every frame)")
    print(f"Detection Confidence: {DETECTION_CONFIDENCE}")
    print(f"Counting Line Y: {line_y}")
    print(f"Video: {video_path}")
    print("=" * 50)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    frame_count = 0
    
    def process_next_frame():
        nonlocal frame_count
        global current_date, fps_counter, fps_start_time, current_fps
        
        ret, frame = cap.read()
        if not ret:
            print("Video processing completed!")
            cap.release()
            return
            
        frame_count += 1
        
        # Skip frames if needed for performance
        if frame_count % (FRAME_SKIP + 1) != 0:
            root.after(1, process_next_frame)  # Process next frame immediately
            return
        
        today = date.today()
        if today != current_date:
            current_date = today
            reset_daily_counts()

        # Run YOLO detection
        results = model.track(
            frame, 
            persist=True, 
            tracker="bytetrack.yaml",
            conf=DETECTION_CONFIDENCE,
            iou=0.5,
            verbose=False
        )

        detection_count = 0
        
        if results[0].boxes.id is not None:
            for box in results[0].boxes:
                class_id = int(box.cls[0].item())
                obj_id = int(box.id.item())

                if 0 <= class_id < len(CATEGORIES):
                    class_name = CATEGORIES[class_id]
                else:
                    continue

                if class_name in ["EMPTY_CONVEYER_BELT"]:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                confidence = float(box.conf[0].item())
                
                detection_count += 1

                # Enhanced line crossing detection
                if enhanced_line_crossing_detection(obj_id, cx, cy, class_name, frame_count):
                    counts_per_class[class_name] += 1
                    counted_ids.add(obj_id)
                    print(f"COUNT UPDATED! {class_name}: {counts_per_class[class_name]}")

                    today_str = today.strftime("%Y-%m-%d")
                    save_daily_count(today_str, class_name, counts_per_class[class_name])
                    update_date_options()
                
                # Update previous_y for next frame comparison
                previous_y[obj_id] = cy

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{class_name} ID:{obj_id} ({confidence:.2f})", 
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                # Draw center point
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        # Draw counting line
        cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (255, 0, 0), 3)
        
        # Draw total count
        total_count = sum(counts_per_class.values())
        cv2.putText(frame, f'Frame: {frame_count}', (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f'Detections: {detection_count}', (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f'Total Count: {total_count}', (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        # Calculate and display FPS
        fps_counter += 1
        current_time = time.time()
        if current_time - fps_start_time >= 1.0:
            current_fps = fps_counter / (current_time - fps_start_time)
            fps_counter = 0
            fps_start_time = current_time
            
        cv2.putText(frame, f'FPS: {current_fps:.1f}', (30, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        # Resize video for display
        frame = cv2.resize(frame, (screen_width, screen_height // 2))
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)

        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

        # Update GUI elements
        live_total_label.config(text=f"Live Total: {total_count}")
        total_label.config(text=f"Total Count: {total_count}")
        fps_label.config(text=f"FPS: {current_fps:.1f}")

        tree.delete(*tree.get_children())
        for cat, count in counts_per_class.items():
            tree.insert('', 'end', values=(cat, count))
        
        # Print status every 50 frames
        if frame_count % 50 == 0:
            print(f"Frame {frame_count}: {detection_count} detections, Total count: {total_count}")

        # Cleanup old tracks periodically
        if frame_count % 30 == 0:
            cleanup_old_tracks(frame_count)
        
        # Schedule next frame processing
        root.after(1, process_next_frame)  # Process next frame as soon as possible
    
    # Start processing
    process_next_frame()

# Start simple detection
root.after(100, run_simple_detection)

# Run the GUI
root.mainloop()