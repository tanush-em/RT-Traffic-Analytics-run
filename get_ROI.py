"""
    USAGE:
        - the video starts playing on default
        - use 'p' to pause or play the video
        - use 'q' to quit the window
        - left click to start drawing the polygon on running video frame
        - right click and 'q' to finish drawing 
        - the script saves the coordinate automatically
"""
# Library Imports and Dependencies
import cv2
import numpy as np

###############################################################################################
# Enter unique camera feed identification name
FILE_NAME = r"" 
# Video path to get ROI coordinates for 
video_path = r""
###############################################################################################
# Variable initialization
roi_points = [] 
drawing = False  
current_frame = None  
target_size = (640, 640)  

def draw_polygon(event, x, y, flags, param):
    """
    Callback function to handle mouse events for drawing the polygon.
    """
    global roi_points, drawing, current_frame

    if event == cv2.EVENT_LBUTTONDOWN:
        # Start drawing the polygon
        drawing = True
        roi_points.append((x, y))
        # Draw a small circle at the point
        cv2.circle(current_frame, (x, y), 5, (0, 255, 0), -1) 
        if len(roi_points) > 1:
            # Draw a line connecting the last two points
            cv2.line(current_frame, roi_points[-2], roi_points[-1], (0, 255, 0), 2)
        cv2.imshow('Get Region of Interest Coordinates', current_frame)  # Show the updated frame

    elif event == cv2.EVENT_RBUTTONDOWN:
        # Finish drawing
        drawing = False
        if len(roi_points) > 2:
            # Complete the polygon by drawing a line to the first point
            cv2.line(current_frame, roi_points[-1], roi_points[0], (0, 255, 0), 2)
            cv2.imshow('Get Region of Interest Coordinates', current_frame)  # Show the updated frame

def resize_with_padding(image, target_size):
    """
    Resize the image to target_size while maintaining the aspect ratio.
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

def select_roi(video_path):
    """
    Function to select the Region of Interest (ROI) in a video.
    """
    global current_frame
    cap = cv2.VideoCapture(video_path)
    paused = False  

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    cv2.namedWindow('Get Region of Interest Coordinates')
    cv2.setMouseCallback('Get Region of Interest Coordinates', draw_polygon) 

    while cap.isOpened():
        if not paused:
            ret, frame = cap.read() 
            if not ret:
                break
            frame = resize_with_padding(frame, target_size)  
            current_frame = frame.copy()  
            cv2.imshow('Get Region of Interest Coordinates', frame)  # Correct window name

        key = cv2.waitKey(30) & 0xFF
        if key == ord('p'):
            paused = not paused
        elif key == ord('q'):
            break

    cap.release() 
    cv2.destroyAllWindows()  
    return np.array(roi_points, np.int32)


# Select ROI and get the points
roi_polygon = select_roi(video_path)
print(f"Selected ROI coordinates: {roi_polygon}")

# Save the ROI coordinates to a file
roi_coordinates_path = r'{}_ROI_coordinates.txt'.format(FILE_NAME)
np.savetxt(roi_coordinates_path, roi_polygon, fmt='%d', delimiter=', ')
print(f"ROI coordinates saved to: {roi_coordinates_path}")
