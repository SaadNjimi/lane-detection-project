import cv2

# gray          → Black and white
# blur          → Smooth edges
# canny         → Sharp white edges
# mask          → Only road area
# HoughLinesP   → Finds straight lines in the edges
# draw lines    → Show the result
# **/
# Load the video
video_path = "challenge.mp4"
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Cannot open video.")
    exit()

while True:
    ret, frame = cap.read()
    
    if not ret:
        break  # End of video

    # Display the frame
    def detect_lanes(frame):
        import numpy as np

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert the frame to grayscale for easier edge detection
        blur = cv2.GaussianBlur(gray, (5, 5), 0) # Apply Gaussian blur to reduce noise and improve edge detection
        edges = cv2.Canny(blur, 50, 150)  # Detect edges using the Canny edge detector

        height, width = edges.shape
        mask = np.zeros_like(edges)
        
        # Define a region of interest (ROI) where we expect the lane lines to be
        polygon = np.array([[
            (0, height),
            (width, height),
            (int(width * 0.55), int(height * 0.6)),
            (int(width * 0.45), int(height * 0.6)),
        ]], np.int32) 
        
        # Mask the image to focus only on the defined ROI
        cv2.fillPoly(mask, polygon, 255) 
        cropped_edges = cv2.bitwise_and(edges, mask)
        
        # Use the Hough Transform to detect lines in the cropped image (shrek points)
        lines = cv2.HoughLinesP(
            cropped_edges,
            rho=2,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=40,
            maxLineGap=100
        ) 

        left_lines = []
        right_lines = []
        
        # Classify the detected lines as left or right based on their slopes
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 - x1 == 0:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                if slope < -0.5:
                    left_lines.append(line[0])
                elif slope > 0.5:
                    right_lines.append(line[0])
        
        # Function to calculate average slope and intercept using linear regression (emla bhal moving average)
        def average_slope_intercept(lines):
            x = []
            y = []
            for x1, y1, x2, y2 in lines:
                x += [x1, x2]
                y += [y1, y2]
            if len(x) == 0:
                return None
            poly = np.polyfit(x, y, 1)
            return poly
        
        # Function to generate a line from a slope-intercept (y = mx + b) form
        def make_line(y1, y2, poly):
            if poly is None:
                return None
            m, b = poly
            x1 = int((y1 - b) / m)
            x2 = int((y2 - b) / m)
            return np.array([x1, y1, x2, y2])
        
        # Create a black image to draw the lane lines on
        line_image = np.zeros_like(frame)
        y1 = height
        y2 = int(height * 0.6)
        
        # Calculate the left and right lane lines by averaging their slopes and intercepts
        # Average all left lines
        left_avg = average_slope_intercept(left_lines)
        left_line = make_line(y1, y2, left_avg)

        # Average all right lines
        right_avg = average_slope_intercept(right_lines)
        right_line = make_line(y1, y2, right_avg)

        # Draw the detected lane lines on the image if they exist
        if left_line is not None:
            cv2.line(line_image, tuple(left_line[:2]), tuple(left_line[2:]), (255, 0, 0), 5)
        else:
            print("Warning: No left lane detected.")

        if right_line is not None:
            cv2.line(line_image, tuple(right_line[:2]), tuple(right_line[2:]), (0, 255, 0), 5)
        else:
            print("Warning: No right lane detected.")


        if left_line is not None:
            cv2.line(line_image, tuple(left_line[:2]), tuple(left_line[2:]), (255, 0, 0), 5)
        if right_line is not None:
            cv2.line(line_image, tuple(right_line[:2]), tuple(right_line[2:]), (0, 255, 0), 5)
        
        # Combine the lane lines with the original frame
        final = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

        # 🧠 STEERING ESTIMATION

        def calculate_steering_angle(frame, left_line, right_line):
            height, width, _ = frame.shape
            if left_line is None or right_line is None:
                return 0  # Go straight if lanes are missing
                        
            # Lane center calculation
            car_center_x = width // 2

            if left_line is not None and right_line is not None:
                mid_x = (left_line[0] + right_line[0]) // 2 # Midpoint between the lanes

            elif left_line is not None:
                # Only left line detected
                mid_x = left_line[0] + 200  # Guess right lane + shift

            elif right_line is not None:
                # Only right line detected
                mid_x = right_line[0] - 200  # Guess left lane - shift

            else:
                # No lanes detected
                mid_x = car_center_x  # Assume straight



            # Calculate the offset from the center of the car
            offset = mid_x - car_center_x

            # Assume the vehicle is moving towards the horizon
            distance_ahead = height / 2  # Approximate distance ahead (half the height of the frame)

            # Calculate the steering angle (in degrees)
            angle_rad = np.arctan2(offset, distance_ahead)
            angle_deg = np.degrees(angle_rad)

            return angle_deg    
        
        # 🧠 Calculate the estimated steering angle
        angle = calculate_steering_angle(frame, left_line, right_line)
        
        # 🖥️ Display the calculated steering angle on the frame
        steering_text = f"Steering Angle: {angle:.2f}°"
        cv2.putText(final, steering_text, (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        print(steering_text)
        
        # Show the final image with detected lane lines and the steering angle
        cv2.imshow("Final Lane Lines + Direction", final)
        return final
    

    # Call it here
    lane_frame = detect_lanes(frame)
    cv2.imshow("Lane Detection", lane_frame)


    # Press 'q' to quit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
print("Video ended. Press any key to close.")
cv2.waitKey(0)  # Wait indefinitely
cv2.destroyAllWindows()
