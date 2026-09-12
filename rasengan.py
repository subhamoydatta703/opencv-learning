import cv2
import mediapipe as mp
import time
import numpy as np

# MediaPipe setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Configure hand landmarker
options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

# Create hand landmarker
landmarker = HandLandmarker.create_from_options(options)

# Load Rasengan
rasengan = cv2.imread(
    "rasengan.png",
    cv2.IMREAD_UNCHANGED
)

# Open webcam
cap = cv2.VideoCapture(0)

# Start timestamp
timestamp = 0

# Start rotation timer
start_time = time.time()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Convert frame to MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Update timestamp
    timestamp += 1

    # Detect hand
    result = landmarker.detect_for_video(
        mp_image,
        timestamp
    )

    # Get first hand
    if result.hand_landmarks:
        landmarks = result.hand_landmarks[0]

        # Select palm landmarks
        palm_indices = [0, 5, 9, 13, 17]

        # Store X and Y coordinates
        xs = []
        ys = []

        for i in palm_indices:
            landmark = landmarks[i]

            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])

            xs.append(x)
            ys.append(y)

        # Calculate palm center
        palm_x = sum(xs) // len(xs)
        palm_y = sum(ys) // len(ys)

        # Measure palm width
        index_knuckle = landmarks[5]
        pinky_knuckle = landmarks[17]

        index_x = index_knuckle.x * frame.shape[1]
        index_y = index_knuckle.y * frame.shape[0]

        pinky_x = pinky_knuckle.x * frame.shape[1]
        pinky_y = pinky_knuckle.y * frame.shape[0]

        palm_width = np.hypot(
            index_x - pinky_x,
            index_y - pinky_y
        )

        # Calculate dynamic Rasengan size
        size = int(
            np.clip(
                palm_width * 2.2,
                80,
                280
            )
        )

        # Resize Rasengan
        rasengan_resized = cv2.resize(
            rasengan,
            (size, size)
        )

        # Calculate smooth rotation
        angle = (time.time() - start_time) * 220

        # Rotate around center
        center = (size // 2, size // 2)

        matrix = cv2.getRotationMatrix2D(
            center,
            angle,
            1.0
        )

        rasengan_resized = cv2.warpAffine(
            rasengan_resized,
            matrix,
            (size, size),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )

        # Calculate overlay position
        x = palm_x - size // 2
        y = palm_y - size // 2

        # Keep overlay inside frame
        if (
            x >= 0
            and y >= 0
            and x + size <= frame.shape[1]
            and y + size <= frame.shape[0]
        ):
            # Get alpha channel
            alpha = rasengan_resized[:, :, 3] / 255.0

            # Blend Rasengan with webcam
            for c in range(3):
                frame[
                    y:y + size,
                    x:x + size,
                    c
                ] = (
                    alpha * rasengan_resized[:, :, c]
                    + (1 - alpha)
                    * frame[y:y + size, x:x + size, c]
                )

    # Show webcam
    cv2.imshow("Rasengan", frame)

    # Exit with ESC
    if cv2.waitKey(1) == 27:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()