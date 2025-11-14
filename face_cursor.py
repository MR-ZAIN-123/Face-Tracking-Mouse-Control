"""
face_cursor.py

Track face position with a laptop webcam to control the OS mouse cursor.
Uses OpenCV's Haar Cascade classifier for face detection.
PyAutoGUI is used to move the system cursor and perform clicks.

Features:
- Cursor follows face center position.
- Blink detection (quick eye closure) to trigger a mouse click.
- Smooth cursor movement and debounce logic to avoid accidental clicks.
- Works with Python 3.13+.

Run: `python face_cursor.py`
"""

import time
import sys
import math

# Try to import required third-party packages
try:
    import cv2
    import numpy as np
    import pyautogui
except Exception as e:
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
SMOOTHING = 0.12           # smoothing factor for cursor movement (0.0 - 1.0)
BLINK_FRAMES = 4           # consecutive frames of blink to trigger click
BLINK_COOLDOWN = 25        # frames to wait before a    llowing next click (debounce)
EYE_AR_THRESHOLD = 0.2     # eye aspect ratio threshold for blink detection

# Joystick-style control settings
JOYSTICK_MODE = True       # enable joystick-style continuous movement
JOYSTICK_DEADZONE = 0.10   # fraction of frame half-width/height considered 'center'
JOYSTICK_MAX_SPEED = 1000  # maximum cursor speed in pixels per second
JOYSTICK_EXPONENT = 1.0    # sensitivity curve exponent (1.0 = linear)

# Load Haar Cascade classifiers (built-in with OpenCV)
FACE_CASCADE = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
EYE_CASCADE = cv2.data.haarcascades + 'haarcascade_eye.xml'

face_cascade = cv2.CascadeClassifier(FACE_CASCADE)
eye_cascade = cv2.CascadeClassifier(EYE_CASCADE)

if face_cascade.empty() or eye_cascade.empty():
    print("ERROR: Could not load Haar Cascade classifiers.")
    sys.exit(1)


def detect_face_center(frame, frame_rgb):
    """
    Detect face in frame and return center coordinates.
    
    Args:
        frame: OpenCV frame in BGR
        frame_rgb: OpenCV frame in grayscale (for detection)
    
    Returns:
        (cx, cy): center coordinates of detected face, or None
    """
    faces = face_cascade.detectMultiScale(frame_rgb, 1.3, 5)
    
    if len(faces) == 0:
        return None
    
    # Use the largest detected face
    largest_face = max(faces, key=lambda f: f[2] * f[3])
    x, y, w, h = largest_face
    
    # Return center of face
    cx = x + w // 2
    cy = y + h // 2
    
    return (cx, cy), (x, y, w, h)


def detect_blink(frame_roi):
    """
    Detect blink using Haar Cascade eye detection.
    If eyes cannot be detected, it likely means a blink/eyes closed.
    
    Args:
        frame_roi: region of interest (face region in grayscale)
    
    Returns:
        blink_detected: True if blink detected (eyes closed/not visible), False otherwise
    """
    if frame_roi is None or frame_roi.size == 0:
        return False
    
    # Try to detect eyes in the face region
    eyes = eye_cascade.detectMultiScale(frame_roi, 1.1, 5, minSize=(15, 15))
    
    # If no eyes detected, assume eyes are closed (blink)
    if len(eyes) == 0:
        return True
    
    # If eyes detected but very small, might be partial blink
    if len(eyes) > 0:
        eye_sizes = [e[2] * e[3] for e in eyes]
        avg_eye_size = np.mean(eye_sizes)
        
        # Small eyes = closed or blinking
        if avg_eye_size < 150:
            return True
    
    return False


def main():
    """
    Main loop: capture video, detect face, track position, detect blink, move cursor, click.
    """
    # Get screen size for mapping coordinates
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

    blink_counter = 0          # consecutive frames of blink detected
    blink_cooldown = 0         # frames remaining before another click allowed
    clicked = False            # whether a click was just sent

    print("Starting face tracking. Press 'q' in the video window to quit.")
    print("Move your head to control cursor. Blink to click.")
    # Time for velocity integration (joystick mode)
    last_time = time.time()

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                time.sleep(0.01)
                continue

            # Mirror the image so movement feels natural
            frame = cv2.flip(frame, 1)
            frame_h, frame_w = frame.shape[:2]

            # Convert to grayscale for face detection
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect face
            result = detect_face_center(frame, frame_gray)

            if result is not None:
                (fx_px, fy_px), face_bbox = result
                x, y, w, h = face_bbox

                # Joystick-style continuous movement: map small face offsets to sustained cursor velocity
                if JOYSTICK_MODE:
                    # Normalized offsets (-1..1) relative to frame center
                    frame_cx = frame_w / 2.0
                    frame_cy = frame_h / 2.0
                    norm_x = (fx_px - frame_cx) / frame_cx
                    norm_y = (fy_px - frame_cy) / frame_cy

                    # Scale normalized offset into velocity (px/sec) with deadzone
                    def axis_velocity(n):
                        if abs(n) <= JOYSTICK_DEADZONE:
                            return 0.0
                        s = (abs(n) - JOYSTICK_DEADZONE) / (1.0 - JOYSTICK_DEADZONE)
                        return math.copysign((s ** JOYSTICK_EXPONENT) * JOYSTICK_MAX_SPEED, n)

                    vx = axis_velocity(norm_x)
                    vy = axis_velocity(norm_y)

                    # Integrate velocity over elapsed time to get cursor delta
                    now = time.time()
                    dt = max(1e-6, now - last_time)
                    last_time = now

                    dx = vx * dt
                    dy = vy * dt

                    new_x = prev_x + dx
                    new_y = prev_y + dy

                    # Clamp to screen bounds
                    new_x = max(0, min(new_x, screen_w - 1))
                    new_y = max(0, min(new_y, screen_h - 1))

                    # Move the OS cursor
                    try:
                        pyautogui.moveTo(new_x, new_y, duration=0)
                    except Exception:
                        pass

                    prev_x, prev_y = new_x, new_y

                    # Draw deadzone rectangle and velocity overlay
                    dzx = int(frame_cx * JOYSTICK_DEADZONE)
                    dzy = int(frame_cy * JOYSTICK_DEADZONE)
                    cx = int(frame_cx)
                    cy = int(frame_cy)
                    cv2.rectangle(frame, (cx - dzx, cy - dzy), (cx + dzx, cy + dzy), (255, 0, 0), 1)
                    cv2.putText(frame, f"V: ({vx:.0f},{vy:.0f})", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                else:
                    # Map pixel coordinates to screen coordinates (direct mapping)
                    target_x = int((fx_px / frame_w) * screen_w)
                    target_y = int((fy_px / frame_h) * screen_h)

                    # Clamp to screen bounds
                    target_x = max(0, min(target_x, screen_w - 1))
                    target_y = max(0, min(target_y, screen_h - 1))

                    # Smooth cursor movement
                    smoothed_x = prev_x + (target_x - prev_x) * SMOOTHING
                    smoothed_y = prev_y + (target_y - prev_y) * SMOOTHING

                    # Move the OS cursor
                    try:
                        pyautogui.moveTo(smoothed_x, smoothed_y, duration=0)
                    except Exception:
                        pass

                    prev_x, prev_y = smoothed_x, smoothed_y

                # Draw face rectangle and center
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, (fx_px, fy_px), 5, (0, 255, 0), -1)

                # Extract face region for blink detection
                face_roi = frame_gray[y:y+h, x:x+w]

                # Detect blink
                blink = detect_blink(face_roi)

                # Handle blink detection + debounce
                if blink:
                    blink_counter += 1
                else:
                    blink_counter = 0

                # Decrement cooldown
                if blink_cooldown > 0:
                    blink_cooldown -= 1

                # If blink held for enough frames and cooldown expired, trigger click
                if blink_counter >= BLINK_FRAMES and blink_cooldown == 0:
                    try:
                        pyautogui.click()
                        print("Click triggered by blink")
                        blink_cooldown = BLINK_COOLDOWN
                        blink_counter = 0  # Reset after click
                    except Exception as e:
                        print(f"Click error: {e}")

                # Draw text (use the current cursor position for display)
                display_x = int(prev_x)
                display_y = int(prev_y)
                cv2.putText(
                    frame,
                    f"Face: ({display_x}, {display_y}) Blink: {blink_counter}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Cooldown: {blink_cooldown}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                )
            else:
                # No face detected
                cv2.putText(
                    frame,
                    "No face detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
                blink_counter = 0

            # Show result frame
            cv2.imshow("Face Cursor Control", frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            # Quit on 'q' key
            if key == ord("q"):
                break
            
            # Click on spacebar press
            if key == ord(" "):
                try:
                    pyautogui.click()
                    print("Click triggered by spacebar")
                except Exception as e:
                    print(f"Click error: {e}")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
