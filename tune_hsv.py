"""
tune_hsv.py

Interactive HSV tuning tool to find the best skin color range for your hand.
Adjust sliders to see real-time mask results, then use the printed values in gesture.py.

Run: python tune_hsv.py
"""

import cv2
import numpy as np

def nothing(x):
    """Callback for trackbar (required but does nothing)."""
    pass

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Create a window for trackbars
    cv2.namedWindow("HSV Tuner")
    
    # Create trackbars for HSV range
    cv2.createTrackbar("Lower H", "HSV Tuner", 0, 180, nothing)
    cv2.createTrackbar("Upper H", "HSV Tuner", 20, 180, nothing)
    cv2.createTrackbar("Lower S", "HSV Tuner", 20, 255, nothing)
    cv2.createTrackbar("Upper S", "HSV Tuner", 255, 255, nothing)
    cv2.createTrackbar("Lower V", "HSV Tuner", 70, 255, nothing)
    cv2.createTrackbar("Upper V", "HSV Tuner", 255, 255, nothing)
    
    print("Adjust trackbars to detect your hand skin tone.")
    print("You should see your hand highlighted in white on the mask.")
    print("Press 'q' to quit and print the final HSV values.")
    print()
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        frame = cv2.flip(frame, 1)
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Read trackbar values
        lower_h = cv2.getTrackbarPos("Lower H", "HSV Tuner")
        upper_h = cv2.getTrackbarPos("Upper H", "HSV Tuner")
        lower_s = cv2.getTrackbarPos("Lower S", "HSV Tuner")
        upper_s = cv2.getTrackbarPos("Upper S", "HSV Tuner")
        lower_v = cv2.getTrackbarPos("Lower V", "HSV Tuner")
        upper_v = cv2.getTrackbarPos("Upper V", "HSV Tuner")
        
        lower_hsv = np.array([lower_h, lower_s, lower_v], dtype=np.uint8)
        upper_hsv = np.array([upper_h, upper_s, upper_v], dtype=np.uint8)
        
        # Create mask
        mask = cv2.inRange(frame_hsv, lower_hsv, upper_hsv)
        
        # Apply morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Display original and mask side-by-side
        frame_small = cv2.resize(frame, (320, 240))
        mask_small = cv2.resize(mask, (320, 240))
        mask_3ch = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
        
        combined = np.hstack([frame_small, mask_3ch])
        cv2.imshow("Original (left) vs Mask (right)", combined)
        
        # Show current values
        cv2.imshow("HSV Tuner", np.zeros((100, 300, 3), dtype=np.uint8))
        cv2.displayOverlay("HSV Tuner", 
            f"Lower: ({lower_h}, {lower_s}, {lower_v})  Upper: ({upper_h}, {upper_s}, {upper_v})", 
            2000)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Print final values for copy-paste into gesture.py
    print()
    print("=" * 60)
    print("FINAL HSV VALUES - Copy these into gesture.py:")
    print("=" * 60)
    print(f"LOWER_SKIN = np.array([{lower_h}, {lower_s}, {lower_v}], dtype=np.uint8)")
    print(f"UPPER_SKIN = np.array([{upper_h}, {upper_s}, {upper_v}], dtype=np.uint8)")
    print("=" * 60)

if __name__ == "__main__":
    main()
