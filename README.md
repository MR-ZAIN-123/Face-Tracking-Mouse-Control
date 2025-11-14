#  Face Tracking Mouse Control

Control your computer's mouse cursor hands-free using your webcam! This Python program uses your laptop webcam to track your face and control the OS mouse cursor with blink detection.

##  Features

-  Real-time face detection using OpenCV Haar Cascade classifiers
-  Blink detection to trigger mouse clicks
-  Joystick-style continuous movement with configurable deadzone
-  Smooth cursor movement following face position
-  Direct coordinate mapping or velocity-based control modes
-  Customizable sensitivity curves for precise control
-  Spacebar as alternative click trigger
-  Debounce logic to prevent accidental clicks

##  Requirements

- Python 3.13+
- A webcam
- OpenCV, NumPy, PyAutoGUI

##  Installation

1. Create and activate a virtual environment (recommended).
2. Install dependencies:

powershell
python -m pip install -r requirements.txt


##  Usage

powershell
python face_cursor.py


##  Controls

- **Move cursor** - Move your head/face
- **Click** - Blink your eyes
- **Spacebar** - Manual click trigger
- **'q' key** - Quit application

##  Configuration & Tuning

Extensive tuning parameters available in ace_cursor.py:

- **SMOOTHING** - Adjust cursor responsiveness (0.0 - 1.0)
- **BLINK_FRAMES** - Consecutive frames of blink to trigger click
- **BLINK_COOLDOWN** - Frames to wait before allowing next click (debounce)
- **JOYSTICK_MODE** - Enable/disable joystick-style movement
- **JOYSTICK_DEADZONE** - Center dead zone fraction
- **JOYSTICK_MAX_SPEED** - Maximum cursor speed in pixels/second
- **JOYSTICK_EXPONENT** - Sensitivity curve exponent

### Tips for Better Detection

- Ensure good lighting on your face
- Position webcam at eye level
- Adjust BLINK_FRAMES if clicks are too sensitive or not responsive
- Tune SMOOTHING if cursor movement is jittery
- On Windows, run the terminal with appropriate privileges for PyAutoGUI to control the mouse

##  Troubleshooting

- **Import errors**: Ensure packages from 
equirements.txt are installed in your active environment
- **Webcam not opening**: Check if another application is using the webcam
- **Poor face detection**: Ensure adequate lighting and position face clearly in frame
- **Cursor lag**: Increase SMOOTHING value for smoother movement

##  Use Cases

- Hands-free computer control
- Accessibility solutions for motor disabilities
- Touchless interaction
- Interactive presentations
- Computer vision demonstrations
