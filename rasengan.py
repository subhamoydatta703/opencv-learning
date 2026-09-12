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

        # Get image dimensions
        height, width = frame.shape[:2]

        # Select palm landmarks
        palm_indices = [0, 5, 9, 13, 17]

        # Store palm coordinates
        xs = []
        ys = []

        for i in palm_indices:
            landmark = landmarks[i]

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            xs.append(x)
            ys.append(y)

        # Calculate palm center
        palm_x = sum(xs) // len(xs)
        palm_y = sum(ys) // len(ys)

        # Measure palm width
        index_knuckle = landmarks[5]
        pinky_knuckle = landmarks[17]

        index_x = index_knuckle.x * width
        index_y = index_knuckle.y * height

        pinky_x = pinky_knuckle.x * width
        pinky_y = pinky_knuckle.y * height

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

        # Calculate palm orientation
        wrist = np.array([
            landmarks[0].x,
            landmarks[0].y,
            landmarks[0].z
        ])

        index_base = np.array([
            landmarks[5].x,
            landmarks[5].y,
            landmarks[5].z
        ])

        pinky_base = np.array([
            landmarks[17].x,
            landmarks[17].y,
            landmarks[17].z
        ])

        vector_a = index_base - wrist
        vector_b = pinky_base - wrist

        palm_normal = np.cross(
            vector_a,
            vector_b
        )

        # Check palm side
        palm_facing_camera = palm_normal[2] < 0

        if palm_facing_camera:

            # Resize Rasengan
            rasengan_resized = cv2.resize(
                rasengan,
                (size, size)
            )

            # Calculate rotation
            angle = (
                time.time() - start_time
            ) * 500

            # Rotate around center
            center = (
                size // 2,
                size // 2
            )

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

            # Create larger glow canvas
            glow_size = int(size * 1.5)
            offset = (
                glow_size - size
            ) // 2

            glow_canvas = np.zeros(
                (glow_size, glow_size, 4),
                dtype=np.uint8
            )

            # Put Rasengan alpha in center
            glow_canvas[
                offset:offset + size,
                offset:offset + size,
                3
            ] = rasengan_resized[:, :, 3]

            # Create glow alpha
            glow_alpha = cv2.GaussianBlur(
                glow_canvas[:, :, 3],
                (0, 0),
                20
            )

            glow_alpha = (
                glow_alpha.astype(np.float32)
                / 255.0
            )

            # Create blue glow
            glow = np.zeros_like(
                glow_canvas
            )

            glow[:, :, 0] = 255
            glow[:, :, 1] = 80
            glow[:, :, 2] = 20

            glow[:, :, 3] = np.clip(
                glow_alpha * 180,
                0,
                255
            ).astype(np.uint8)

            # Calculate overlay position
            x = palm_x - glow_size // 2
            y = palm_y - glow_size // 2

            # Check overlay bounds
            if (
                x >= 0
                and y >= 0
                and x + glow_size <= width
                and y + glow_size <= height
            ):

                # Get glow alpha
                glow_alpha = (
                    glow[:, :, 3] / 255.0
                )

                # Blend glow
                for c in range(3):
                    frame[
                        y:y + glow_size,
                        x:x + glow_size,
                        c
                    ] = (
                        glow_alpha
                        * glow[:, :, c]
                        + (1 - glow_alpha)
                        * frame[
                            y:y + glow_size,
                            x:x + glow_size,
                            c
                        ]
                    )

                # Calculate Rasengan position
                rx = x + offset
                ry = y + offset

                # Get Rasengan alpha
                alpha = (
                    rasengan_resized[:, :, 3]
                    / 255.0
                )

                # Blend Rasengan
                for c in range(3):
                    frame[
                        ry:ry + size,
                        rx:rx + size,
                        c
                    ] = (
                        alpha
                        * rasengan_resized[:, :, c]
                        + (1 - alpha)
                        * frame[
                            ry:ry + size,
                            rx:rx + size,
                            c
                        ]
                    )

    # Show webcam
    cv2.imshow(
        "Rasengan",
        frame
    )

    # Exit with ESC
    if cv2.waitKey(1) == 27:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()