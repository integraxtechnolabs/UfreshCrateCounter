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

# ----------------- YOLO + ByteTrack Setup -----------------
model = YOLO("E:/best.pt")  # change to your model path
video_path = "output_2025-04-01_15-21-09.mp4"

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

# ----------------- Tkinter GUI Setup -----------------
root = tk.Tk()
root.title("Crate Counter")

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

live_total_label = tk.Label(
    tab1, 
    text="Live Total: 0", 
    font=("Arial", max(14, screen_height // 40)), 
    fg="green"
)
live_total_label.pack(pady=10)

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
    global counts_per_class, counted_ids, previous_y
    counts_per_class = {cat: 0 for cat in CATEGORIES}
    counted_ids.clear()
    previous_y.clear()
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

# ----------------- Video Processing Thread -----------------
def run_detection():
    global current_date
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        today = date.today()
        if today != current_date:
            current_date = today
            reset_daily_counts()

        results = model.track(frame, persist=True, tracker="bytetrack.yaml")

        # if results[0].boxes.id is not None:
        #     for box in results[0].boxes:
        #         class_id = int(box.cls[0].item())
        #         obj_id = int(box.id.item())

        #         if 0 <= class_id < len(CATEGORIES):
        #             class_name = CATEGORIES[class_id]
        #         else:
        #             continue
        #         if class_name in ["EMPTY_CONVEYER_BELT"]:
        #             continue

        #         x1, y1, x2, y2 = map(int, box.xyxy[0])
        #         cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        #         if obj_id in previous_y:
        #             prev_y = previous_y[obj_id]
        #             if prev_y > line_y and cy <= line_y and obj_id not in counted_ids:
        #                 counts_per_class[class_name] += 1
        #                 counted_ids.add(obj_id)

        #                 today_str = today.strftime("%Y-%m-%d")
        #                 save_daily_count(today_str, class_name, counts_per_class[class_name])
        #                 update_date_options()

        #         previous_y[obj_id] = cy
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

                # --- Draw bounding box ---
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{class_name} ID:{obj_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                # --- Draw center point of crate ---
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)   # red filled circle
                cv2.putText(frame, f"({cx},{cy})", (cx + 10, cy), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

                # --- Line crossing logic ---
                if obj_id in previous_y:
                    prev_y = previous_y[obj_id]
                    if prev_y > line_y and cy <= line_y and obj_id not in counted_ids:
                        counts_per_class[class_name] += 1
                        counted_ids.add(obj_id)

                        today_str = today.strftime("%Y-%m-%d")
                        save_daily_count(today_str, class_name, counts_per_class[class_name])
                        update_date_options()

                previous_y[obj_id] = cy

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{class_name} ID:{obj_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (255, 0, 0), 2)
        total_count = sum(counts_per_class.values())
        cv2.putText(frame, f'Total Count: {total_count}', (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        # Resize video dynamically to fit screen
        frame = cv2.resize(frame, (screen_width, screen_height // 2))
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)

        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

        live_total_label.config(text=f"Live Total: {total_count}")
        total_label.config(text=f"Total Count: {total_count}")

        tree.delete(*tree.get_children())
        for cat, count in counts_per_class.items():
            tree.insert('', 'end', values=(cat, count))

    cap.release()

# ----------------- Start Detection -----------------
threading.Thread(target=run_detection, daemon=True).start()
root.mainloop()