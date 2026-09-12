import os
import time
import math
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading face landmark model to {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done.")
    return MODEL_PATH



LEFT_IRIS = [468, 469, 470, 471]
RIGHT_IRIS = [473, 474, 475, 476]

LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]


def get_iris_center_and_radius(landmarks, idxs, w, h):
    pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in idxs], dtype=np.float32)
    center = pts.mean(axis=0)
    radius = float(np.mean(np.linalg.norm(pts - center, axis=1)))
    return center, radius


def eye_mask(shape, landmarks, idxs, w, h, feather=7):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in idxs], dtype=np.int32)
    hull = cv2.convexHull(pts)
    cv2.fillConvexPoly(mask, hull, 255)
    if feather > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), feather)
    return mask


def tomoe_polygon(center, orbit_radius, size, angle_deg, n=24):
    theta = np.linspace(0, math.radians(230), n)
    frac = theta / theta.max()
    r = orbit_radius * (1 - frac * 0.9)
    bulge = size * np.sin(np.pi * frac) * 0.6

    local_x = r * np.cos(theta) - bulge * 0.2
    local_y = r * np.sin(theta)
    pts = np.stack([local_x, local_y], axis=1)

    ang = math.radians(angle_deg)
    rot = np.array([[math.cos(ang), -math.sin(ang)],
                    [math.sin(ang), math.cos(ang)]])
    pts = pts @ rot.T + center

    head = (center[0] + orbit_radius * math.cos(ang),
            center[1] + orbit_radius * math.sin(ang))
    return pts.astype(np.int32), (int(head[0]), int(head[1]))


def draw_sharingan(frame, center, radius, angle_deg,
                    iris_color=(20, 20, 200), ring_color=(0, 0, 0)):
    overlay = frame.copy()
    cx, cy = int(center[0]), int(center[1])
    r = int(radius * 1.05)

    cv2.circle(overlay, (cx, cy), r, iris_color, -1, cv2.LINE_AA)
    cv2.circle(overlay, (cx, cy), r, ring_color, max(2, r // 8), cv2.LINE_AA)

    for k in range(3):
        a = angle_deg + k * 120
        pts, head = tomoe_polygon(center, r * 0.62, r * 0.30, a)
        cv2.fillPoly(overlay, [pts], ring_color, cv2.LINE_AA)
        cv2.circle(overlay, head, max(2, int(r * 0.16)), ring_color, -1, cv2.LINE_AA)

    cv2.circle(overlay, (cx, cy), max(2, int(r * 0.22)), ring_color, -1, cv2.LINE_AA)
    cv2.circle(overlay, (cx - r // 4, cy - r // 4), max(1, r // 10), (60, 60, 60), -1, cv2.LINE_AA)

    return overlay


def main():
    model_path = ensure_model()

    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    start = time.time()

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.time() - start) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        angle_deg = (time.time() - start) * 90 % 360  # constant rotation speed

        if result.face_landmarks:
            lm = result.face_landmarks[0]  # list of landmarks with .x/.y/.z
            out = frame.copy()

            for iris_idx, eye_idx in ((LEFT_IRIS, LEFT_EYE), (RIGHT_IRIS, RIGHT_EYE)):
                center, radius = get_iris_center_and_radius(lm, iris_idx, w, h)
                if radius < 2:
                    continue
                sharingan = draw_sharingan(frame, center, radius, angle_deg)
                mask = eye_mask(frame.shape, lm, eye_idx, w, h)
                mask3 = cv2.merge([mask, mask, mask]).astype(np.float32) / 255.0
                out = (out.astype(np.float32) * (1 - mask3) +
                       sharingan.astype(np.float32) * mask3).astype(np.uint8)

            frame = out

        cv2.imshow("Sharingan Effect", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()