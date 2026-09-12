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

rasengan = cv2.imread("rasengan.png", cv2.IMREAD_UNCHANGED)

# open webcam
cap = cv2.VideoCapture(0)

# timestamp for frame
timestamp = 0

# angle for rasengan rotation
angle =0

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

        #resize rasengan

        size = 150

        rasengan_resized = cv2.resize(rasengan, (size, size))

        # update rasengan rotation
        angle += 5

        # rotate rasengan around center
        center = (size//2, size//2)

        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        rasengan_resized = cv2.warpAffine(rasengan_resized, matrix, (size, size), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
        

        # overlay position
        x = palm_x-size //2
        y = palm_y-size //2

        #overlay inside frame
        if(x>= 0 and y>=0 and x+ size < frame.shape[1] and y + size <= frame.shape[0]):
            alpha = rasengan_resized[:, :, 3] / 255.0
            # overlay rasengan
            for c in range(3):
                frame[y:y + size, x:x + size, c]=(alpha * rasengan_resized[:, :, c] + (1- alpha) * frame[y:y + size, x:x + size, c])


        # cv2.circle( frame, (palm_x, palm_y), 5, (0, 255, 0), -1)


    # show webcam
    cv2.imshow("Rasengan", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
