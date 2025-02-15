# RT-Traffic-Analytics-run
**This is a proprietary project and any suspection of copying of IP wil lead to legal consequences.**
RT-Traffic-Analytics is a real-time traffic analysis tool that utilizes a YOLOv8 model for object detection and the DeepSORT algorithm for object tracking. 
**This repository is dedicated to ONLY running the model and its services.**
This project is designed to analyze video footage and count unique vehicles, categorized into different classes such as cars, buses, two-wheelers, heavy vehicles, and auto-rickshaws.

## Features
    - Real-time Object Detection: Uses a YOLOv8 model trained on a custom dataset for detecting different classes of vehicles
    - Object Tracking: Integrates the DeepSORT tracking algorithm to uniquely identify and count vehicles across video frames.
    - Output Video with Annotations: Generates an output video with bounding boxes and labels for detected and tracked objects.
    - Object Counting: Maintains a count of unique objects detected throughout the video.
    - Output Statistics: Saves the object counts in a separate file.
   
## Requirements
    - Python 3.10.12
    - YOLOv8 (Ultralytics)
    - deep-sort-realtime (Deep Simple Online and Realtime Tracking)
    - OpenCV
    - Torch and other libraries
   
## Workflow
    - Before setting up the project it is highly recommended to go through the docs
    - Use the *get_ROI.py* file to get the coordinates of the focus polygon
    - Save the coordinates in *camera_profiles.xlxs*
    - Now run *app.py*
