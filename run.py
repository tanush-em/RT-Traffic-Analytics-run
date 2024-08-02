# Import libraries and packages
import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Path to pretrained weight '.pt' file
WEIGHTS_PATH = r""

# Define paths for input video and output video
VIDEO_PATH = r""
OUTPUT_VIDEO_PATH = r""
COUNT_FILE_PATH = r""

# Tuneable Parameters
# Define Region of Interest (ROI) coordinates
ROI_PX_COORDINATES = np.array([[64, 267], [89, 383], [307, 322], [197, 273]], np.int32) # Define ROI coordinates from get_ROI.py here
# Frame skip rate for reducing computation
FRAME_SKIP_RATE = 1
# Confidence threshold for model prediction
PREDICTION_CONFIDENCE_THRESHOLD = 0.4

# Loading the model
model = YOLO(WEIGHTS_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

target_size = (640, 640)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, target_size)

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30, n_init=3)
object_counts = defaultdict(set)

def is_within_roi(box, ROI_PX_COORDINATES):
    """
    Check if the center of the bounding box is within the ROI polygon.
    """
    x1, y1, x2, y2 = map(int, box)
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    return cv2.pointPolygonTest(ROI_PX_COORDINATES, (center_x, center_y), False) >= 0

def resize_with_padding(image, target_size):
    """
    Resize the image with padding to match the target size.
    """
    original_h, original_w = image.shape[:2]
    target_h, target_w = target_size

    scaling_factor = min(target_w / original_w, target_h / original_h)
    new_w = int(original_w * scaling_factor)
    new_h = int(original_h * scaling_factor)

    resized_image = cv2.resize(image, (new_w, new_h))

    new_image = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    new_image.fill(0)

    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2

    new_image[top:top + new_h, left:left + new_w] = resized_image

    return new_image

frame_counter = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1
    if frame_counter % FRAME_SKIP_RATE != 0:
        continue

    # Resize the frame with padding
    frame = resize_with_padding(frame, target_size)

    # Perform object detection using YOLO model
    results = model(frame)
    detections = []
    classes = []

    # Extract bounding boxes, confidences, and class labels
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        cls_array = result.boxes.cls.cpu().numpy()

        for box, confidence, cls in zip(boxes, confidences, cls_array):
            if confidence > PREDICTION_CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = map(int, box)
                label = model.names[int(cls)]

                if is_within_roi(box, ROI_PX_COORDINATES):
                    width = x2 - x1
                    height = y2 - y1
                    detections.append(([x1, y1, width, height], float(confidence), int(cls)))
                    classes.append(int(cls))
                    color = (0, 255, 0)  # Green bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Update the tracker with current frame's detections
    tracks = tracker.update_tracks(detections, frame=frame, others=classes)

    # Draw tracking information and update object counts
    for track, cls in zip(tracks, classes):
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        label = model.names[cls]

        # Draw bounding box and track ID
        color = (255, 0, 0)  # Blue for tracking
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        cv2.putText(frame, f'{label}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Update the count if the object is within ROI for the first time
        if is_within_roi((x1, y1, x2, y2), ROI_PX_COORDINATES):
            if track_id not in object_counts[label]:
                object_counts[label].add(track_id)
                
    cv2.polylines(frame, [ROI_PX_COORDINATES], isClosed=True, color=(255, 0, 0), thickness=1)
    out.write(frame)
cap.release()
out.release()

# Print the object counts summary
object_counts_summary = {label: len(ids) for label, ids in object_counts.items()}
print("Object Counts:", object_counts_summary)

# Save the object counts to a text file
with open(COUNT_FILE_PATH, 'w') as f:
    for label, count in object_counts_summary.items():
        f.write(f'{label}: {count}\n')


# Load the YOLO model with pretrained weights
model = YOLO('/content/drive/MyDrive/100epochs-best.pt')

# Define paths for input video and output video
VIDEO_PATH = r'/content/drive/MyDrive/CC_TV_Video/NVR_ch40_main_20240724110000_20240724110017.mp4'
OUTPUT_VIDEO_PATH = r'/content/sample.mp4'

# Tuneable Parameters
# Define Region of Interest (ROI) coordinates
ROI_PX_COORDINATES = np.array([[64, 267], [89, 383], [307, 322], [197, 273]], np.int32)
# Frame skip rate for reducing computation
FRAME_SKIP_RATE = 2
# Confidence threshold for model prediction
PREDICTION_CONFIDENCE_THRESHOLD = 0.4

# Capture video from the specified path
cap = cv2.VideoCapture(VIDEO_PATH)

# Retrieve video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Set target dimensions for video frames
target_size = (640, 640)

# Define codec and create VideoWriter object for the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, target_size)

# Initialize the Deep SORT tracker
tracker = DeepSort(max_age=30, n_init=3)

# Dictionary to keep track of unique object counts
object_counts = defaultdict(set)

def is_within_roi(box, ROI_PX_COORDINATES):
    """
    Check if the center of the bounding box is within the ROI polygon.
    """
    x1, y1, x2, y2 = map(int, box)
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    return cv2.pointPolygonTest(ROI_PX_COORDINATES, (center_x, center_y), False) >= 0

def resize_with_padding(image, target_size):
    """
    Resize the image with padding to match the target size.
    """
    original_h, original_w = image.shape[:2]
    target_h, target_w = target_size

    scaling_factor = min(target_w / original_w, target_h / original_h)
    new_w = int(original_w * scaling_factor)
    new_h = int(original_h * scaling_factor)

    resized_image = cv2.resize(image, (new_w, new_h))

    new_image = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    new_image.fill(0)

    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2

    new_image[top:top + new_h, left:left + new_w] = resized_image

    return new_image

frame_counter = 0
# Process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1
    if frame_counter % FRAME_SKIP_RATE != 0:
        continue

    # Resize the frame with padding
    frame = resize_with_padding(frame, target_size)

    # Perform object detection using YOLO model
    results = model(frame)
    detections = []
    classes = []

    # Extract bounding boxes, confidences, and class labels
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        cls_array = result.boxes.cls.cpu().numpy()

        for box, confidence, cls in zip(boxes, confidences, cls_array):
            if confidence > PREDICTION_CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = map(int, box)
                label = model.names[int(cls)]

                if is_within_roi(box, ROI_PX_COORDINATES):
                    width = x2 - x1
                    height = y2 - y1
                    detections.append(([x1, y1, width, height], float(confidence), int(cls)))
                    classes.append(int(cls))
                    color = (0, 255, 0)  # Green bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Update the tracker with current frame's detections
    tracks = tracker.update_tracks(detections, frame=frame, others=classes)

    # Draw tracking information and update object counts
    for track, cls in zip(tracks, classes):
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        label = model.names[cls]

        # Draw bounding box and track ID
        color = (255, 0, 0)  # Blue for tracking
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        cv2.putText(frame, f'{label}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Update the count if the object is within ROI for the first time
        if is_within_roi((x1, y1, x2, y2), ROI_PX_COORDINATES):
            if track_id not in object_counts[label]:
                object_counts[label].add(track_id)

    # Draw the ROI on the frame
    cv2.polylines(frame, [ROI_PX_COORDINATES], isClosed=True, color=(255, 0, 0), thickness=1)

    # Write the processed frame to the output video file
    out.write(frame)

# Release video capture and writer objects
cap.release()
out.release()

# Print the object counts summary
object_counts_summary = {label: len(ids) for label, ids in object_counts.items()}
print("Object Counts:", object_counts_summary)

# Save the object counts to a text file
with open('/content/object_counts.txt', 'w') as f:
    for label, count in object_counts_summary.items():
        f.write(f'{label}: {count}\n')
