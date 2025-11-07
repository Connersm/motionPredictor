import cv2
import numpy as np
from collections import deque
import pandas as pd
import time

motion_log = []

def initialize_capture():
    vid = cv2.VideoCapture(0)
    if not vid.isOpened():
        raise RuntimeError("Could not open video source.")
    return vid

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (5, 5), 0)

def get_temporal_diff(gray, prev_gray, threshold=20):
    if prev_gray is None:
        return None, gray
    diff = cv2.absdiff(gray, prev_gray)
    _, diff_mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    return diff_mask, gray

def get_background_mask(frame, back_sub, threshold=200):
    fg_mask = back_sub.apply(frame)
    _, fg_mask = cv2.threshold(fg_mask, threshold, 255, cv2.THRESH_BINARY)
    return fg_mask

def combine_and_clean_masks(diff_mask, fg_mask):
    combined = cv2.bitwise_and(fg_mask, diff_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=3)
    combined = cv2.dilate(combined, None, iterations=2)
    return combined

def merge_contours(mask, area_threshold=1500):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour_mask = np.zeros_like(mask)
    cv2.drawContours(contour_mask, contours, -1, 255, -1)
    merged_contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in merged_contours:
        if cv2.contourArea(c) < area_threshold:
            continue
        boxes.append(cv2.boundingRect(c))
    return boxes

def smooth_boxes(box_history, current_boxes, history_size=5):
    if len(box_history) >= history_size:
        box_history.popleft()
    box_history.append(current_boxes)
    all_boxes = [b for frame_boxes in box_history for b in frame_boxes]
    if not all_boxes:
        return []
    xs = [b[0] for b in all_boxes]
    ys = [b[1] for b in all_boxes]
    ws = [b[2] for b in all_boxes]
    hs = [b[3] for b in all_boxes]
    avg_box = (int(np.mean(xs)), int(np.mean(ys)), int(np.mean(ws)), int(np.mean(hs)))
    return [avg_box]

def extract_object_data(boxes, prev_center, prev_time):
    if not boxes:
        return None, prev_center, prev_time
    (x, y, w, h) = boxes[0]
    cx, cy = x + w // 2, y + h // 2
    area = w * h
    now = time.time()
    if prev_center is not None and prev_time is not None:
        dt = now - prev_time
        vx = (cx - prev_center[0]) / dt
        vy = (cy - prev_center[1]) / dt
    else:
        vx = vy = 0
    return (cx, cy, vx, vy, area), (cx, cy), now

def update_motion_history(motion_log, data):
    if data is not None:
        motion_log.append(data)

def save_motion_data(motion_log, filename="motion_data.csv"):
    if not motion_log:
        return
    df = pd.DataFrame(motion_log, columns=["time", "x", "y", "vx", "vy", "area"])
    df.to_csv(filename, index=False)

def draw_boxes(frame, boxes):
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Object", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

def encode_frame(frame):
    ret, buffer = cv2.imencode(".jpg", frame)
    if not ret:
        return None
    return (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

def read_vid():
    vid = initialize_capture()
    back_sub = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=50, detectShadows=False)
    prev_gray = None
    frame_skip = 2
    frame_count = 0
    box_history = deque()
    history_size = 5
    motion_log = []
    prev_center = None
    prev_time = None

    while True:
        ret, frame = vid.read()
        if not ret or frame is None:
            break
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue
        gray = preprocess_frame(frame)
        diff_mask, prev_gray = get_temporal_diff(gray, prev_gray)

        if diff_mask is None:
            continue

        fg_mask = get_background_mask(frame, back_sub)
        combined_mask = combine_and_clean_masks(diff_mask, fg_mask)

        boxes = merge_contours(combined_mask)
        smoothed_boxes = smooth_boxes(box_history, boxes, history_size)

        data, prev_center, prev_time = extract_object_data(smoothed_boxes, prev_center, prev_time)

        if data is not None:
            timestamp = time.time()
            update_motion_history(motion_log, (timestamp, data[0], data[1], data[2], data[3], data[4]))
        
        draw_boxes(frame, smoothed_boxes)
        encoded = encode_frame(frame)
        if encoded:
            yield encoded
    vid.release()
