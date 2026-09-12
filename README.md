# OpenCV Webcam Projects

Small Python projects for experimenting with OpenCV webcam capture and face detection.

## Projects

- `webcam.py` - Displays the live webcam feed.
- `face-detection.py` - Detects faces using the Haar Cascade classifier.
- `face-detection-dnn.py` - Detects faces using OpenCV's YuNet DNN detector.
- `cv2-ver.py` - Prints the installed OpenCV version and feature information.

## Requirements

- Python 3.12 or newer
- A working webcam
- OpenCV Python package

The repository includes the model files required by the face-detection examples:

- `haarcascade_frontalface_default.xml`
- `face_detection_yunet_2026may.onnx`

## Setup

From the project root, create or activate a virtual environment. For the existing
`.venv2` environment in PowerShell:

```powershell
.\.venv2\Scripts\Activate.ps1
```

Install OpenCV:

```powershell
python -m pip install opencv-python
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
```

Press `Esc` to close the webcam window. If the camera does not open, check that
another application is not using it and try changing `cv2.VideoCapture(0)` to
another camera index such as `1`.

## Check the installation

```powershell
python .\cv2-ver.py
```
