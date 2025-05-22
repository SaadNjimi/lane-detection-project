# Lane Detection & Steering Angle Estimation 🚗🛣️

This project implements a lane detection system using OpenCV to identify road lanes from video frames and estimate the steering angle needed to keep the vehicle centered in the lane.

## Features

- Grayscale conversion & Gaussian blur for noise reduction  
- Canny edge detection to identify edges in the image  
- Region of Interest (ROI) masking to focus on the road area  
- Hough Transform to detect lane lines  
- Average slope/intercept calculation to smooth lane lines  
- Steering angle estimation based on lane position relative to the car center  
- Real-time visualization of detected lanes and steering angle  

## How It Works

1. Convert the frame to grayscale and apply Gaussian blur to smooth the image.  
2. Use Canny edge detection to highlight lane edges.  
3. Mask the image to focus only on the road region.  
4. Detect lines with the Hough Transform and classify them as left or right lanes.  
5. Average the detected lines to create single left and right lane lines.  
6. Calculate the steering angle from the lane center relative to the vehicle center.  
7. Overlay the detected lanes and steering angle on the original frame.  

## Usage

1. Make sure you have OpenCV and NumPy installed:  
```bash
pip install opencv-python numpy
Run the Python script:

bash
Copy
Edit
python lane_detection.py
Put your video file (challenge.mp4) in the same folder or update the path in the script.

Press q to quit the video window.
