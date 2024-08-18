# Library and Package imports
import os
import streamlit as st
import pandas as pd
import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch

# Load camera profiles from an Excel file
camera_profiles_df = pd.read_excel('camera_profiles.xlsx')  # Update with your file path

# Path to the pre-trained YOLO model weights
MODEL_WEIGHTS_PATH = r"E:\Infomaps\RT Vehicle Detection\build\RT-Traffic-Analytics-run\weights\100epochs-day.pt"

# Directory to temporarily store uploaded videos
TEMP_VIDEO_DIR = "temp/"

# Ensure that the directory exists
os.makedirs(TEMP_VIDEO_DIR, exist_ok=True)

def get_camera_profile_details(profile_id):
    """
    Retrieves the location and ROI coordinates for a specified camera profile ID.

    Args:
        profile_id (int): The ID of the camera profile.

    Returns:
        location (str): The location associated with the camera profile.
        roi_coordinates (np.ndarray): The ROI coordinates as a NumPy array of tuples.
    """
    row = camera_profiles_df[camera_profiles_df['Camera_Profile_ID'] == profile_id].iloc[0]
    location = row['Location']
    roi_coordinates = np.array([tuple(map(int, coord.split(','))) for coord in row['ROI_Coordinates'].split(' ')])
    return location, roi_coordinates

def resize_with_padding(image, target_size=(640, 640)):
    """
    Resizes an image to the specified target size while maintaining the aspect ratio by adding padding.

    Args:
        image (np.ndarray): The input image to resize.
        target_size (tuple): The desired target size as (height, width).

    Returns:
        new_image (np.ndarray): The resized image with padding.
    """
    original_h, original_w = image.shape[:2]
    target_h, target_w = target_size
    scaling_factor = min(target_w / original_w, target_h / original_h)
    new_w = int(original_w * scaling_factor)
    new_h = int(original_h * scaling_factor)
    resized_image = cv2.resize(image, (new_w, new_h))
    new_image = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    # Calculate padding to center the image
    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    new_image[top:top + new_h, left:left + new_w] = resized_image
    return new_image

def process_video(video_path, roi_px_coordinates, MODEL_WEIGHTS_PATH, DEVICE='cuda', frame_skip_rate=1, confidence_threshold=0.5):
    """
    Processes a video and counts objects within a defined ROI using a YOLO model and DeepSort tracker.

    Args:
        video_path (str): The path to the video file.
        roi_px_coordinates (np.ndarray): The coordinates defining the ROI.
        MODEL_WEIGHTS_PATH (str): The path to the YOLO model weights.
        DEVICE (str): The device to run the model on ('cuda' or 'cpu').
        frame_skip_rate (int): The number of frames to skip during processing.
        confidence_threshold (float): The minimum confidence required for detections.

    Returns:
        object_counts_summary (dict): A dictionary containing the count of each detected object within the ROI.
    """
    # Load YOLO model
    model = YOLO(MODEL_WEIGHTS_PATH).to(DEVICE)
    cap = cv2.VideoCapture(video_path)
    target_size = (640, 640)  # Resize target size
    tracker = DeepSort(max_age=20, n_init=3)  # Initialize DeepSort tracker
    object_counts = defaultdict(set)  # Dictionary to store object counts

    # Function to check if a detected object is within the defined ROI
    def is_within_roi(box, roi_px_coordinates):
        x1, y1, x2, y2 = map(int, box)
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        return cv2.pointPolygonTest(roi_px_coordinates, (center_x, center_y), False) >= 0

    frame_counter = 0  # Counter to track the frames

    # Process each frame of the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1
        if frame_counter % frame_skip_rate != 0:
            continue  # Skip frames based on the frame skip rate

        # Resize the frame
        frame = resize_with_padding(frame, target_size)
        
        # Convert frame to tensor for model input
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)

        # Get detection results from YOLO model
        results = model(frame_tensor)
        detections = []
        classes = []

        # Process each detection result
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            cls_array = result.boxes.cls.cpu().numpy()

            # Filter detections within the ROI and above the confidence threshold
            for box, confidence, cls in zip(boxes, confidences, cls_array):
                if confidence >= confidence_threshold and is_within_roi(box, roi_px_coordinates):
                    x1, y1, x2, y2 = map(int, box)
                    width = x2 - x1
                    height = y2 - y1
                    detections.append(([x1, y1, width, height], float(confidence), int(cls)))
                    classes.append(int(cls))

        # Update tracker with current frame's detections
        tracks = tracker.update_tracks(detections, frame=frame, others=classes)

        # Track and count objects within the ROI
        for track, cls in zip(tracks, classes):
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)

            if is_within_roi((x1, y1, x2, y2), roi_px_coordinates):
                label = model.names[cls]
                if track_id not in object_counts[label]:
                    object_counts[label].add(track_id)

    cap.release()  # Release video capture object

    # Summarize object counts
    object_counts_summary = {label: len(ids) for label, ids in object_counts.items()}
    return object_counts_summary

# Streamlit UI setup
st.title("Real-Time Traffic Analytics")

# Sidebar controls
st.sidebar.title("CONTROLS")
device_option = st.sidebar.selectbox("Select Device", ('CPU', 'CUDA'))
DEVICE = 'cuda' if device_option == 'CUDA' else 'cpu'

# Camera profile selection
camera_profile_id = st.sidebar.selectbox("Select Camera Profile", camera_profiles_df['Camera_Profile_ID'])
location, roi_coordinates = get_camera_profile_details(camera_profile_id)

# Display video details
st.subheader(f"Video Details")
st.write(f"Camera Profile ID: {camera_profile_id}")
st.write(f"Location: {location}")
st.write(f"ROI Coordinates: {roi_coordinates}")

# Video file uploader
uploaded_video = st.sidebar.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

if uploaded_video is not None:
    # Define video path in the temporary directory
    video_path = os.path.join(TEMP_VIDEO_DIR, uploaded_video.name)
    
    # Save the uploaded video to the temporary directory
    with open(video_path, mode='wb') as f:
        f.write(uploaded_video.read())

    # Display the first frame with ROI overlay
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if ret:
        resized_frame = resize_with_padding(frame)
        roi_image = cv2.polylines(resized_frame.copy(), [roi_coordinates], isClosed=True, color=(0, 255, 0), thickness=2)
        st.image(roi_image, channels="BGR", caption="Resized Video Preview with ROI")
    cap.release()

    # Process video button
    if st.button("Process Video"):
        with st.spinner('Processing Video...'):
            object_counts = process_video(video_path, roi_coordinates, MODEL_WEIGHTS_PATH, DEVICE=DEVICE, frame_skip_rate=1, confidence_threshold=0.3)
        st.success("Processing Complete!")
        st.write("Object Counts:")
        st.json(object_counts)

        # Delete the video file after processing
        try:
            os.remove(video_path) 
            st.write("Temporary file successfully deleted.")
        except Exception as e:
            st.error(f"Error deleting temporary file: {e}")

else:
    st.sidebar.warning("Please upload a video file.")