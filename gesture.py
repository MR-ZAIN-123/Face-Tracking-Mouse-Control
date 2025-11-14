"""
gesture.py

Track fingertip motion with a laptop webcam and control the OS mouse.
Uses OpenCV color-space hand detection (HSV skin detection) + contour analysis.
PyAutoGUI is used to move the system cursor and perform clicks.

Features:
- Cursor follows the detected fingertip (highest point of hand contour).
- Pinch gesture (closed fist or tight hand) triggers a click.
- Smooth cursor movement and debounce logic to avoid accidental clicks.
- Works with Python 3.13+ (no MediaPipe required).

Run: `python gesture.py`
"""

import time
import math
import sys

# Try to import required third-party packages and provide a helpful message
# if they are missing so the user knows how to install them.
try:
    import cv2
    import numpy as np
    import pyautogui
except Exception as e:  # ModuleNotFoundError or other import issues
    print("ERROR: Failed to import required packages:", e)
    print()
    print("Install the required packages into your Python environment:")
    print(r"python -m pip install -r \"c:\\Users\\ZAIN\\finger gesture\\requirements.txt\"")
    print()
    print("Or install individually:")
    print(r"python -m pip install opencv-python numpy pyautogui")
    sys.exit(1)


# ----------------------------- Configuration -----------------------------
# Tweak these values to change responsiveness and sensitivity.
SMOOTHING = 0.15           # smoothing factor for cursor movement (0.0 - 1.0)
PINCH_THRESHOLD = 0.35     # closure ratio threshold to detect closed fist (pinch)
PINCH_FRAMES = 8           # number of consecutive frames pinch must hold to click
RELEASE_FRAMES = 5         # frames of release required to reset click state
HAND_AREA_MIN = 3000       # minimum contour area to consider as a hand (lowered)
HAND_AREA_MAX = 250000     # maximum contour area to consider as a hand (raised)

# HSV color range for skin detection
# If hand detection is poor, run tune_hsv.py to find better values for your skin tone
LOWER_SKIN = np.array([0, 10, 60], dtype=np.uint8)
UPPER_SKIN = np.array([25, 255, 255], dtype=np.uint8)

# Blur and edge detection parameters
BLUR_SIZE = 9              # kernel size for Gaussian blur (odd number)
CANNY_LOW = 50             # lower threshold for Canny edge detection
CANNY_HIGH = 150           # upper threshold for Canny edge detection


def detect_hand_contour(frame_hsv, frame_bgr):
    """
    Detect skin-colored regions in HSV space with multiple fallback strategies.
    
    Args:
        frame_hsv: OpenCV frame in HSV color space
        frame_bgr: OpenCV frame in BGR color space (for edge detection fallback)
    
    Returns:
        contour: largest hand-like contour or None if no suitable contour found
    """
    # Primary: HSV skin-color mask
    mask = cv2.inRange(frame_hsv, LOWER_SKIN, UPPER_SKIN)
    
    # Apply morphological operations to reduce noise and connect nearby regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Dilate to fill small holes
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    # Fallback: If HSV mask is too empty, use edge-based detection
    if cv2.countNonZero(mask) < 500:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (BLUR_SIZE, BLUR_SIZE), 0)
        edges = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)
        mask = cv2.dilate(edges, kernel, iterations=3)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Filter contours by area and select the largest one
    valid_contours = [c for c in contours 
                      if HAND_AREA_MIN < cv2.contourArea(c) < HAND_AREA_MAX]
    
    if not valid_contours:
        return None
    
    # Return the contour with the largest area (assumed to be the hand)
    largest_contour = max(valid_contours, key=cv2.contourArea)
    return largest_contour


def get_fingertip_position(contour, frame_shape):
    """
    Extract the fingertip position from a hand contour by finding the highest point
    (topmost in the frame, which is typically the fingertip when hand is raised).
    
    Args:
        contour: OpenCV contour
        frame_shape: shape of the frame (height, width, channels)
    
    Returns:
        (x, y): pixel coordinates of the detected fingertip, or None if contour is invalid
    """
    if contour is None or len(contour) < 5:
        return None
    
    # Find the topmost point (minimum y coordinate) - this is typically the fingertip
    topmost = tuple(contour[contour[:, :, 1].argmin()][0])
    return topmost


def compute_hand_closure(contour):
    """
    Estimate how closed the hand is (0 = open, 1 = closed fist) by comparing
    the contour area to its bounding rectangle area.
    
    Args:
        contour: OpenCV contour
    
    Returns:
        closure: float in [0, 1] representing how closed the hand is
    """
    if contour is None:
        return 0.0
    
    area = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)
    rect_area = w * h
    
    if rect_area == 0:
        return 0.0
    
    # Closure ratio: lower values = more open, higher values = more closed
    closure = 1.0 - (area / rect_area)
    return closure


def main():
    """
    Main loop: capture video, detect hand, track fingertip, detect pinch, move cursor, click.
    """
    # Get screen size for mapping coordinates to screen pixels
    screen_w, screen_h = pyautogui.size()

    # Initialize webcam capture
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        sys.exit(1)

    # Set capture resolution
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    prev_x, prev_y = screen_w // 2, screen_h // 2

    pinch_counter = 0        # counts consecutive frames where pinch is detected
    release_counter = 0      # counts consecutive frames where pinch is released
    clicked = False          # whether a click has been sent and waiting for release

    print("Starting hand tracking. Press 'q' in the video window to quit.")

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                time.sleep(0.01)
                continue

            # Mirror the image so movement feels natural (like a mirror).
            frame = cv2.flip(frame, 1)
            frame_h, frame_w = frame.shape[:2]

            # Convert frame to HSV color space (better for skin detection)
            frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Detect hand contour (pass both HSV and BGR for fallback detection)
            hand_contour = detect_hand_contour(frame_hsv, frame)

            if hand_contour is not None:
                # Get the fingertip position (highest point of contour)
                fingertip = get_fingertip_position(hand_contour, frame.shape)

                if fingertip is not None:
                    fx_px, fy_px = fingertip

                    # Map pixel coordinates to screen coordinates
                    target_x = int((fx_px / frame_w) * screen_w)
                    target_y = int((fy_px / frame_h) * screen_h)

                    # Clamp to screen bounds
                    target_x = max(0, min(target_x, screen_w - 1))
                    target_y = max(0, min(target_y, screen_h - 1))

                    # Smooth cursor movement
                    smoothed_x = prev_x + (target_x - prev_x) * SMOOTHING
                    smoothed_y = prev_y + (target_y - prev_y) * SMOOTHING

                    # Move the OS cursor to the smoothed position
                    try:
                        pyautogui.moveTo(smoothed_x, smoothed_y, duration=0)
                    except Exception:
                        # In some environments pyautogui may throw; ignore and continue.
                        pass

                    prev_x, prev_y = smoothed_x, smoothed_y

                    # Compute hand closure to detect pinch/fist
                    closure = compute_hand_closure(hand_contour)

                    # Visual feedback: draw contour and fingertip
                    cv2.drawContours(frame, [hand_contour], 0, (0, 255, 0), 2)
                    cv2.circle(frame, (fx_px, fy_px), 8, (0, 255, 0), -1)

                    # Handle pinch detection + debounce logic
                    # Pinch is detected when hand closure exceeds threshold (hand is closing/closed)
                    if closure > PINCH_THRESHOLD:
                        pinch_counter += 1
                        release_counter = 0
                    else:
                        release_counter += 1
                        pinch_counter = 0

                    # If pinch held for enough consecutive frames and we haven't clicked yet, click
                    if pinch_counter >= PINCH_FRAMES and not clicked:
                        try:
                            pyautogui.click()
                            clicked = True
                        except Exception:
                            # Ignore click errors (e.g. permission issues)
                            pass

                    # Only reset clicked after pinch is released for a few frames
                    if clicked and release_counter >= RELEASE_FRAMES:
                        clicked = False

                    # Draw text showing closure and click state
                    cv2.putText(
                        frame,
                        f"closure={closure:.2f} clicked={clicked}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                    )
                    cv2.putText(
                        frame,
                        f"pinch_counter={pinch_counter} release_counter={release_counter}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        2,
                    )
            else:
                # No hand detected
                cv2.putText(
                    frame,
                    "No hand detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            # Show result frame
            cv2.imshow("Finger Mouse", frame)

            # Quit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
