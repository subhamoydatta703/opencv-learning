import cv2
import mediapipe as mp

# MediaPipe setup
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Configure landmarker
options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="face_landmarker.task"
    ),
    running_mode=VisionRunningMode.IMAGE
)

# Create landmarker
landmarker = FaceLandmarker.create_from_options(options)

# Read image
image = cv2.imread("varun_dhawan.jpg")

# Check image
if image is None:
    print("Could not load image.")
    exit()

# Convert image for MediaPipe
mp_image = mp.Image(
    image_format=mp.ImageFormat.SRGB,
    data=image
)

# Detect face landmarks
result = landmarker.detect(mp_image)

# Check face
if not result.face_landmarks:
    print("No face detected.")
    exit()

# Get first face
landmarks = result.face_landmarks[0]

# Get image size
height, width = image.shape[:2]

# Left iris landmarks
left_iris_indices = [473, 474, 475, 476, 477]

# Convert iris points to pixels
iris_points = []

for index in left_iris_indices:
    landmark = landmarks[index]

    x = int(landmark.x * width)
    y = int(landmark.y * height)

    iris_points.append((x, y))

# Get iris boundaries
xs = [point[0] for point in iris_points]
ys = [point[1] for point in iris_points]

left = min(xs)
right = max(xs)
top = min(ys)
bottom = max(ys)

# Calculate iris size
iris_width = right - left
iris_height = bottom - top

# Calculate iris center
iris_center_x = (left + right) // 2
iris_center_y = (top + bottom) // 2

# Calculate Sharingan size
sharingan_size = int(max(iris_width, iris_height) * 1.2)

# Draw iris points
for x, y in iris_points:
    cv2.circle(image, (x, y), 3, (0, 255, 0), -1)

# Draw iris center
cv2.circle(
    image,
    (iris_center_x, iris_center_y),
    5,
    (0, 0, 255),
    -1
)

# Display image
cv2.imshow("Left Iris", image)
cv2.waitKey(0)
cv2.destroyAllWindows()