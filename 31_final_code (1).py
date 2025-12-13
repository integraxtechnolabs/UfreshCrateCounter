import cv2
import threading
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from ultralytics import YOLO
import csv
import os
from datetime import date
import pandas as pd
import queue
import time

# ----------------- CONFIG -----------------
MODEL_PATH = "E:/best.pt"        # update if needed
video_path = "E:/output_2025-04-01_15-06-09.mp4"                 # 0 = default camera, or path to file

# Inference resolution (smaller = faster). Keep aspect ratio roughly same as camera.
INFER_W, INFER_H = 320, 320    # safe, fast. Try 416x416 if you want slightly better accuracy.

# Display size (what you show in the GUI). Keep reasonably small for speed.
DISPLAY_W, DISPLAY_H = 640, 360

# How often GUI polls results (ms). Lower = smoother but more CPU on main thread.
GUI_POLL_MS = 80               # ~12.5 FPS GUI update

# Only update the category table every N GUI updates
TABLE_UPDATE_EVERY = 2

# Line crossing y-coordinate in display coordinate system; set relative to DISPLAY_H
LINE_Y_DISPLAY = int(DISPLAY_H * 0.5)

# CSV settings
CSV_FILE = "daily_counts.csv"

# ----------------- YOLO + ByteTrack -----------------
model = YOLO(MODEL_PATH)

# Categories - keep as before
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

# ----------------- STATE -----------------
# counts, tracking sets
counts_per_class = {cat: 0 for cat in CATEGORIES}
counted_ids = set()
previous_y = {}  # stores previous center y (in infer coords) keyed by obj id
current_date = date.today()

# CSV init
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "Category", "Count"])

# Only save CSV when a particular category count increased
last_saved_counts = counts_per_class.copy()

# Queue for thread-safe transfer to GUI
gui_queue = queue.Queue(maxsize=2)  # keep latest frames only

# ----------------- Tkinter GUI Setup -----------------
root = tk.Tk()
root.title("Crate Counter")

# fixed display geometry (you can keep fullscreen if you prefer)
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
# Use a reasonable window size so GUI rendering is fast
root.geometry(f"{DISPLAY_W}x{DISPLAY_H+160}+0+0")

notebook = ttk.Notebook(root)
notebook.pack(fill="both", expand=True)

# --- Tab 1: Live Tracking ---
tab1 = ttk.Frame(notebook)
notebook.add(tab1, text="Live Tracking")

video_label = tk.Label(tab1)
video_label.pack()

live_total_label = tk.Label(
    tab1,
    text="Live Total: 0",
    font=("Arial", 14),
    fg="green"
)
live_total_label.pack(pady=5)

# --- Tab 2: Category Counts ---
tab2 = ttk.Frame(notebook)
notebook.add(tab2, text="Category Counts")

total_label = tk.Label(
    tab2,
    text="Total Count: 0",
    font=("Arial", 14),
    fg="blue"
)
total_label.pack(pady=5)

tree = ttk.Treeview(tab2, columns=("Category", "Count"), show='headings', height=12)
tree.heading("Category", text="Category")
tree.heading("Count", text="Count")
tree.pack(fill="both", expand=True, padx=5, pady=5)

# Create tree rows once and keep their ids
tree_item_ids = {}
for cat in CATEGORIES:
    iid = tree.insert('', 'end', values=(cat, 0))
    tree_item_ids[cat] = iid

# --- Tab 3: Daily Summary ---
frame3 = ttk.Frame(notebook)
notebook.add(frame3, text="Daily Summary")

date_label = ttk.Label(frame3, text="Select Date:", font=("Arial", 12))
date_label.pack(pady=5)

date_combo = ttk.Combobox(frame3, state="readonly", font=("Arial", 11))
date_combo.pack(pady=5)

summary_tree = ttk.Treeview(frame3, columns=("Category", "Count"), show="headings")
summary_tree.heading("Category", text="Category")
summary_tree.heading("Count", text="Count")
summary_tree.pack(fill="both", expand=True, padx=10, pady=10)

# ----------------- Helper Functions -----------------
def reset_daily_counts():
    global counts_per_class, counted_ids, previous_y, last_saved_counts
    counts_per_class = {cat: 0 for cat in CATEGORIES}
    counted_ids.clear()
    previous_y.clear()
    last_saved_counts = counts_per_class.copy()
    live_total_label.config(text="Live Total: 0")
    total_label.config(text="Total Count: 0")
    for cat in CATEGORIES:
        tree.item(tree_item_ids[cat], values=(cat, 0))

def update_date_options():
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        if "Date" in df.columns:
            unique_dates = sorted(df["Date"].astype(str).unique())
            date_combo["values"] = unique_dates

def show_summary_for_date(event=None):
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

def save_daily_count_if_changed(today_str, category):
    """Save to CSV only if count changed since last save for that category"""
    global last_saved_counts
    new_count = counts_per_class.get(category, 0)
    if last_saved_counts.get(category, -1) != new_count:
        # load, update or append
        if os.path.exists(CSV_FILE):
            df = pd.read_csv(CSV_FILE)
            if df.empty or "Date" not in df.columns:
                df = pd.DataFrame(columns=["Date", "Category", "Count"])
        else:
            df = pd.DataFrame(columns=["Date", "Category", "Count"])
        mask = (df["Date"] == today_str) & (df["Category"] == category)
        if mask.any():
            df.loc[mask, "Count"] = new_count
        else:
            df = pd.concat([df, pd.DataFrame([[today_str, category, new_count]],
                                             columns=["Date", "Category", "Count"])],
                           ignore_index=True)
        df.to_csv(CSV_FILE, index=False)
        last_saved_counts[category] = new_count
        # refresh date options in case new date added
        update_date_options()

# ----------------- Video / Detection Thread -----------------
def run_detection():
    global current_date, counts_per_class, previous_y, counted_ids

    cap = cv2.VideoCapture(video_path)
    # Try to set capture for speed (may fail for some cameras/files)
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    except Exception:
        pass

    frame_id = 0
    while True:
        ret, full_frame = cap.read()
        if not ret:
            # End of video or camera error - push a final update and exit
            try:
                gui_queue.put_nowait(("eof", None))
            except queue.Full:
                pass
            break

        # Resize frame for display (keep aspect ratio)
        display_frame = cv2.resize(full_frame, (DISPLAY_W, DISPLAY_H))

        # Prepare smaller frame for inference (speed)
        infer_frame = cv2.resize(full_frame, (INFER_W, INFER_H))

        today = date.today()
        if today != current_date:
            current_date = today
            reset_daily_counts()

        # Run tracking on the smaller frame - specify imgsz equal to INFER size
        try:
            results = model.track(infer_frame, persist=True, tracker="bytetrack.yaml", imgsz=max(INFER_W, INFER_H), verbose=False)
        except Exception as e:
            # If tracking crashes, push an error frame but continue
            print("Tracking error:", e)
            results = None

        # Prepare draw on display_frame by mapping boxes from infer coords -> display coords
        if results and len(results) > 0 and getattr(results[0], "boxes", None) is not None:
            # get scaling factor from infer -> display
            # we used full_frame -> infer_frame and full_frame -> display_frame. To map infer->display:
            h_full, w_full = full_frame.shape[:2]
            sx = DISPLAY_W / INFER_W
            sy = DISPLAY_H / INFER_H

            boxes = results[0].boxes  # Boxes object
            # ensure boxes have ids
            if getattr(boxes, "id", None) is not None:
                for box in boxes:
                    try:
                        class_id = int(box.cls[0].item())
                        obj_id = int(box.id.item())
                    except Exception:
                        continue

                    if not (0 <= class_id < len(CATEGORIES)):
                        continue
                    class_name = CATEGORIES[class_id]
                    if class_name == "EMPTY_CONVEYER_BELT":
                        continue

                    # box.xyxy is in infer coords
                    x1_i, y1_i, x2_i, y2_i = map(int, box.xyxy[0].tolist())
                    # center in infer coords
                    cx_i = (x1_i + x2_i) // 2
                    cy_i = (y1_i + y2_i) // 2

                    # line crossing logic needs to be consistent - use infer Y for counts
                    prev_y = previous_y.get(obj_id, None)
                    if prev_y is not None:
                        # If object moved from below the line to above the line in infer coords
                        # Map display line y to infer coordinates for comparison:
                        infer_line_y = int(LINE_Y_DISPLAY / sy)
                        if prev_y > infer_line_y and cy_i <= infer_line_y and obj_id not in counted_ids:
                            counts_per_class[class_name] += 1
                            counted_ids.add(obj_id)
                            today_str = today.strftime("%Y-%m-%d")
                            save_daily_count_if_changed(today_str, class_name)
                    previous_y[obj_id] = cy_i

                    # Map box coords to display coords to draw
                    x1_d = int(x1_i * sx)
                    y1_d = int(y1_i * sy)
                    x2_d = int(x2_i * sx)
                    y2_d = int(y2_i * sy)
                    cx_d = int(cx_i * sx)
                    cy_d = int(cy_i * sy)

                    # draw on display_frame (no double-draw)
                    cv2.rectangle(display_frame, (x1_d, y1_d), (x2_d, y2_d), (0, 255, 0), 2)
                    cv2.putText(display_frame, f"{class_name} ID:{obj_id}", (x1_d, y1_d - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
                    cv2.circle(display_frame, (cx_d, cy_d), 4, (0, 0, 255), -1)
            else:
                # If boxes have no ids, we can't reliably count; skip counting logic
                pass

        # draw line and total count on display
        total_count = sum(counts_per_class.values())
        cv2.line(display_frame, (0, LINE_Y_DISPLAY), (DISPLAY_W, LINE_Y_DISPLAY), (255, 0, 0), 2)
        cv2.putText(display_frame, f'Total Count: {total_count}', (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Convert BGR -> RGB -> PIL -> ImageTk for Tkinter
        rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=pil_img)

        # Put latest frame and stats into GUI queue (non-blocking)
        try:
            # keep only latest: if full, remove one then put
            if gui_queue.full():
                try:
                    _ = gui_queue.get_nowait()
                except Exception:
                    pass
            gui_queue.put_nowait(("frame", imgtk, total_count, counts_per_class.copy()))
        except queue.Full:
            pass

        frame_id += 1
        # small sleep to yield; adjust if you want to throttle
        time.sleep(0.01)

    cap.release()

# ----------------- GUI Polling -----------------
gui_update_counter = 0

def gui_poll():
    global gui_update_counter
    try:
        while not gui_queue.empty():
            item = gui_queue.get_nowait()
            if item[0] == "eof":
                # end of stream
                return
            _, imgtk, total_count, counts_snapshot = item
            # update image
            video_label.imgtk = imgtk
            video_label.configure(image=imgtk)

            # update totals
            live_total_label.config(text=f"Live Total: {total_count}")
            total_label.config(text=f"Total Count: {total_count}")

            # update tree every N GUI updates
            gui_update_counter += 1
            if gui_update_counter % TABLE_UPDATE_EVERY == 0:
                for cat, count in counts_snapshot.items():
                    tree.item(tree_item_ids[cat], values=(cat, count))

    except Exception as e:
        print("GUI poll error:", e)

    # schedule next poll
    root.after(GUI_POLL_MS, gui_poll)

# ----------------- Start Detection Thread and GUI -----------------
det_thread = threading.Thread(target=run_detection, daemon=True)
det_thread.start()

# Start polling loop
root.after(GUI_POLL_MS, gui_poll)
root.mainloop()
