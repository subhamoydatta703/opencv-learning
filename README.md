# OpenCV and MediaPipe Vision Projects

Small Python projects for experimenting with webcam capture, face and hand
landmarks, and anime-inspired visual effects.

## Projects

- `webcam.py` - Displays the live webcam feed.
- `face-detection.py` - Detects faces using the Haar Cascade classifier.
- `face-detection-dnn.py` - Detects faces using OpenCV's YuNet DNN detector.
- `mediapie-learn.py` - Finds iris landmarks in `varun_dhawan.jpg` with MediaPipe.
- `sharingan-effect.py` - Demonstrates YuNet face and eye-coordinate detection.
- `sharingan_effect.py` - Applies a live Sharingan-style effect to the webcam feed.
- `rasengan.py` - Tracks one hand with MediaPipe and overlays a rotating Rasengan effect on the palm.


## Requirements

- Python 3.12 or newer
- A working webcam
- `opencv-python`
- `numpy`
- `mediapipe` for the face-landmark, hand-landmark, and effect examples

The repository includes the model files and image used by the examples:

- `haarcascade_frontalface_default.xml`
- `face_detection_yunet_2026may.onnx`
- `face_landmarker.task`
- `hand_landmarker.task`
- `rasengan.png`
- `varun_dhawan.jpg`

## Setup

From the project root, create or activate a virtual environment. For the existing
`.venv2` environment in PowerShell:

```powershell
.\.venv2\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install opencv-python numpy mediapipe
```

You can also use the existing `venv` environment by activating it with:

```powershell
.\venv\Scripts\Activate.ps1
```

## Run an example

```powershell
python .\webcam.py
python .\face-detection.py
python .\face-detection-dnn.py
python .\mediapie-learn.py
python .\sharingan-effect.py
python .\sharingan_effect.py
python .\rasengan.py
```

Press `Esc` to close the webcam window in the OpenCV examples. Press `q` to
close the `sharingan_effect.py` window. The Rasengan effect also closes with
`Esc`; it needs `rasengan.png` with a transparent background. If the camera does
not open, check that another application is not using it and try changing
`cv2.VideoCapture(0)` to another camera index such as `1`.

## MediaPipe on Windows

MediaPipe loads a native file named `libmediapipe.dll`. On Windows 11, Smart App
Control or a work/school application-control policy can block that DLL and raise
`WinError 4551`. If this is a managed device, ask the administrator to allow the
approved MediaPipe package. On a personal device, only change Smart App Control
settings if you trust the package and understand that it reduces a Windows
security layer.

## Check the installation

```powershell
python .\cv2-ver.py
```
