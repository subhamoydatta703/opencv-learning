import cv2
import mediapipe as mp

# mediapipe setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# configure hand landmarker
options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

# create hand landmarker
landmarker = HandLandmarker.create_from_options(options)

# open webcam
cap = cv2.VideoCapture(0)

# timestamp for frame
timestamp = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # convert to MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # increase timestamp
    timestamp += 1

    # detect hand
    result = landmarker.detect_for_video(
        mp_image,
        timestamp
    )

    # get first hand and landmarks
    if result.hand_landmarks:
        landmarks = result.hand_landmarks[0]

        #palm indices
        palm_indices = [0,5,9,13,17]
        # store xs and ys
        xs=[]
        ys=[]
        for i in palm_indices:
            landmark= landmarks[i]
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])

            xs.append(x)
            ys.append(y)

        #palm_center calculate
        palm_x = sum(xs) // len(xs)
        palm_y = sum(ys) // len(ys)

        cv2.circle( frame, (palm_x, palm_y), 5, (0, 255, 0), -1)


    # show webcam
    cv2.imshow("Rasengan", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
