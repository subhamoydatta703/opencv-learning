import cv2
import mediapipe as mp
import time
import numpy as np
import math

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

landmarker = HandLandmarker.create_from_options(options)
cap = cv2.VideoCapture(0)

timestamp = 0
start_time = time.time()
previous_time = start_time

noise_cache = None
grid_cache = {}

charge_time = 0.0
CHARGE_DURATION = 1.6
CHARGE_DRAIN_RATE = 2.5
MAX_HAND_DISTANCE = 260

FORMING_GRACE_PERIOD = 0.35
TRACKING_SMOOTHING = 0.35

MIN_SIZE = 70
MAX_SIZE = 280

last_mid_x = 0
last_mid_y = 0
last_average_palm_width = 90.0
last_hand_distance = 130.0
lost_detection_timer = 0.0

rng = np.random.default_rng()

RASENGAN_IMAGE_PATH = "rasengan.png"
RASENGAN_IMAGE_OPACITY = 0.35

_rasengan_image_cache = None
_rasengan_image_missing_warned = False

RIM_LIGHT_STRENGTH = 0.55
SPECULAR_STRENGTH = 0.5
TENDRIL_STRENGTH = 0.35
REFRACTION_STRENGTH = 5.0


def get_rasengan_image():
    global _rasengan_image_cache, _rasengan_image_missing_warned

    if _rasengan_image_cache is None and RASENGAN_IMAGE_PATH:
        img = cv2.imread(RASENGAN_IMAGE_PATH, cv2.IMREAD_UNCHANGED)

        if img is None:
            if not _rasengan_image_missing_warned:
                print(f"[rasengan] could not load '{RASENGAN_IMAGE_PATH}', using procedural texture only")
                _rasengan_image_missing_warned = True
            return None

        _rasengan_image_cache = img

    return _rasengan_image_cache


def get_noise_texture(n=512):
    global noise_cache

    if noise_cache is None:
        base_rng = np.random.default_rng(3)
        noise = base_rng.random((n, n)).astype(np.float32)
        noise = cv2.GaussianBlur(noise, (0, 0), 3.0)
        noise_cache = cv2.normalize(noise, None, 0, 1, cv2.NORM_MINMAX)

    return noise_cache


def get_grid(size):
    if size not in grid_cache:
        cx = size / 2
        cy = size / 2

        ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
        dx = xs - cx
        dy = ys - cy
        r = np.sqrt(dx * dx + dy * dy)
        theta = np.arctan2(dy, dx)

        grid_cache[size] = (dx, dy, r, theta)

        if len(grid_cache) > 60:
            grid_cache.pop(next(iter(grid_cache)))

    return grid_cache[size]


def sphere_radius(size):
    return size * 0.42


def get_palm_info(landmarks, width, height):
    palm_indices = [0, 5, 9, 13, 17]
    xs = [int(landmarks[i].x * width) for i in palm_indices]
    ys = [int(landmarks[i].y * height) for i in palm_indices]

    palm_x = sum(xs) // len(xs)
    palm_y = sum(ys) // len(ys)

    index_knuckle = landmarks[5]
    pinky_knuckle = landmarks[17]

    palm_width = np.hypot(
        index_knuckle.x * width - pinky_knuckle.x * width,
        index_knuckle.y * height - pinky_knuckle.y * height
    )

    wrist = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])
    index_base = np.array([landmarks[5].x, landmarks[5].y, landmarks[5].z])
    pinky_base = np.array([landmarks[17].x, landmarks[17].y, landmarks[17].z])

    palm_normal = np.cross(index_base - wrist, pinky_base - wrist)
    palm_facing_camera = palm_normal[2] < 0

    return palm_x, palm_y, palm_width, palm_facing_camera


def build_rasengan_texture(size, t, charge_fraction, jitter):

    dx, dy, r, theta = get_grid(size)
    R = sphere_radius(size)

    instability = 1.0 - charge_fraction
    speed_scale = 1.0 + charge_fraction * 1.6

    # Combine counter-rotating spiral blades.
    l1 = np.sin(5 * theta - r * 0.10 + t * 4.0 * speed_scale)
    l2 = np.sin(3 * theta + r * 0.06 - t * 2.6 * speed_scale)
    l3 = np.sin(8 * theta - r * 0.18 + t * 6.5 * speed_scale) * 0.5

    noise = get_noise_texture()
    nh, nw = noise.shape
    ang = t * 1.7 * speed_scale
    scale = (nw / size) * 1.4

    rxs = (nw / 2) + (dx * math.cos(ang) - dy * math.sin(ang)) * scale
    rys = (nh / 2) + (dx * math.sin(ang) + dy * math.cos(ang)) * scale

    warped_noise = cv2.remap(noise, rxs, rys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    turbulence_weight = 0.16 + instability * 0.18
    combined = 0.55 * l1 + 0.4 * l2 + 0.3 * l3 + turbulence_weight * (warped_noise - 0.5)
    combined = cv2.normalize(combined, None, 0, 1, cv2.NORM_MINMAX)
    combined = np.clip(combined + jitter, 0, 1) ** 1.8

    edge_sharpness = 3.2 - instability * 0.7
    falloff = np.clip(1.0 - (r / R) ** edge_sharpness, 0, 1)
    edge_fade = np.clip((R * (1.05 + instability * 0.10) - r) / (R * 0.10), 0, 1)
    falloff = falloff * edge_fade

    intensity = combined * falloff

    # Add a bright center core.
    core = np.clip(1.3 - r / (R * 0.30), 0, 1) ** 2.2

    # Add orbiting energy wisps.
    orbit_radius = R * 0.68
    squash = 0.6
    wisp_count = 4
    ring_speed = 3.0 * speed_scale
    ring_glow = np.zeros_like(intensity)

    for i in range(wisp_count):
        angle = (t * ring_speed) + (i * (2 * math.pi / wisp_count))
        px = orbit_radius * math.cos(angle)
        py = orbit_radius * math.sin(angle) * squash

        tangent_x = -math.sin(angle)
        tangent_y = math.cos(angle) * squash

        rel_x = dx - px
        rel_y = dy - py
        along = rel_x * tangent_x + rel_y * tangent_y
        across = rel_x * (-tangent_y) + rel_y * tangent_x

        sigma_along = size * 0.09
        sigma_across = size * 0.028
        brightness = 0.5 + 0.4 * math.sin(angle + i)

        streak = np.exp(-((along ** 2) / (2 * sigma_along ** 2) + (across ** 2) / (2 * sigma_across ** 2)))
        ring_glow = ring_glow + brightness * streak

    ring_glow = np.clip(ring_glow, 0, 1.1) * (0.5 + 0.5 * charge_fraction)

    # Brighten the outer rim.
    r_norm = r / R
    rim = np.clip((r_norm - 0.78) / 0.22, 0, 1) * falloff * RIM_LIGHT_STRENGTH

    # Add a glossy highlight for spherical depth.
    hl_x, hl_y = -R * 0.32, -R * 0.38
    hl_sigma = size * 0.07
    highlight = np.exp(-(((dx - hl_x) ** 2) / (2 * hl_sigma ** 2) + ((dy - hl_y) ** 2) / (2 * (hl_sigma * 0.85) ** 2)))
    highlight = highlight * falloff * SPECULAR_STRENGTH

    # Add small tendrils near the outer shell.
    tendril_mask = np.clip((warped_noise - 0.82) / 0.08, 0, 1)
    tendril_band = np.clip((r_norm - 0.55) / 0.45, 0, 1)
    tendril = tendril_mask * tendril_band * falloff * TENDRIL_STRENGTH

    glow_term = core + ring_glow + rim + highlight + tendril

    b = np.clip(255 * (0.6 + 0.4 * intensity) + 140 * glow_term, 0, 255)
    g = np.clip(200 * intensity + 220 * glow_term, 0, 255)
    rr = np.clip(50 * intensity + 130 * glow_term, 0, 255)
    a = np.clip(255 * falloff * (0.6 + 0.4 * combined) + 180 * (ring_glow + rim), 0, 255)

    texture = np.dstack([b, g, rr, a]).astype(np.float32)

    overlay = get_rasengan_image()

    if overlay is not None:
        resized = cv2.resize(overlay, (size, size), interpolation=cv2.INTER_AREA)

        rot_deg = math.degrees(t * 1.7 * speed_scale)
        rot_matrix = cv2.getRotationMatrix2D((size / 2, size / 2), rot_deg, 1.0)
        has_alpha = resized.ndim == 3 and resized.shape[2] == 4

        rotated = cv2.warpAffine(
            resized, rot_matrix, (size, size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0) if has_alpha else (0, 0, 0)
        )

        if has_alpha:
            overlay_bgr = rotated[:, :, :3].astype(np.float32)
            overlay_alpha = rotated[:, :, 3].astype(np.float32) / 255.0
        else:
            overlay_bgr = rotated.astype(np.float32)
            overlay_alpha = np.ones((size, size), dtype=np.float32)

        blend = np.minimum(overlay_alpha, falloff) * RASENGAN_IMAGE_OPACITY
        blend = blend[:, :, None]
        texture[:, :, :3] = texture[:, :, :3] * (1 - blend) + overlay_bgr * blend

    return texture


def composite_rasengan(frame, x, y, texture):

    size = texture.shape[0]
    height, width = frame.shape[:2]

    x0 = max(x, 0)
    y0 = max(y, 0)
    x1 = min(x + size, width)
    y1 = min(y + size, height)

    if x1 <= x0 or y1 <= y0:
        return frame

    tx0 = x0 - x
    ty0 = y0 - y
    tx1 = tx0 + (x1 - x0)
    ty1 = ty0 + (y1 - y0)

    tex_crop = texture[ty0:ty1, tx0:tx1]
    color = tex_crop[:, :, :3]
    alpha = tex_crop[:, :, 3:4] / 255.0

    # Refract the background slightly to avoid a flat-sticker appearance.
    dx, dy, r, _ = get_grid(size)
    R = sphere_radius(size)

    dx_c = dx[ty0:ty1, tx0:tx1]
    dy_c = dy[ty0:ty1, tx0:tx1]
    r_c = np.clip(r[ty0:ty1, tx0:tx1], 1e-3, None)

    u = np.clip(r_c / R, 0, 1)
    lens = REFRACTION_STRENGTH * u * (1 - u) * 4

    nx = dx_c / r_c
    ny = dy_c / r_c

    region_h = y1 - y0
    region_w = x1 - x0

    base_x = np.arange(x0, x1, dtype=np.float32)[None, :]
    base_y = np.arange(y0, y1, dtype=np.float32)[:, None]

    map_x = (np.broadcast_to(base_x, (region_h, region_w)) + nx * lens).astype(np.float32)
    map_y = (np.broadcast_to(base_y, (region_h, region_w)) + ny * lens).astype(np.float32)

    background = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    bloom = cv2.GaussianBlur(color * alpha, (0, 0), size * 0.025)

    region = background.astype(np.float32)
    region = region + bloom * 0.25
    region = region + color * alpha

    frame[y0:y1, x0:x1] = np.clip(region, 0, 255).astype(np.uint8)

    return frame


while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    timestamp += 1
    result = landmarker.detect_for_video(mp_image, timestamp)

    now = time.time()
    dt = now - previous_time
    previous_time = now

    forming = False

    if result.hand_landmarks and len(result.hand_landmarks) >= 2:
        height, width = frame.shape[:2]

        first_hand = result.hand_landmarks[0]
        second_hand = result.hand_landmarks[1]

        palm_x_a, palm_y_a, palm_width_a, facing_a = get_palm_info(first_hand, width, height)
        palm_x_b, palm_y_b, palm_width_b, facing_b = get_palm_info(second_hand, width, height)

        hand_distance = np.hypot(palm_x_a - palm_x_b, palm_y_a - palm_y_b)

        # Track position only while both palms face the camera.
        if facing_a and facing_b and hand_distance < MAX_HAND_DISTANCE:

            raw_mid_x = (palm_x_a + palm_x_b) // 2
            raw_mid_y = (palm_y_a + palm_y_b) // 2
            raw_palm_width = (palm_width_a + palm_width_b) / 2

            if last_mid_x == 0 and last_mid_y == 0:
                last_mid_x = raw_mid_x
                last_mid_y = raw_mid_y
                last_average_palm_width = raw_palm_width
                last_hand_distance = hand_distance
            else:
                last_mid_x = int(last_mid_x + (raw_mid_x - last_mid_x) * TRACKING_SMOOTHING)
                last_mid_y = int(last_mid_y + (raw_mid_y - last_mid_y) * TRACKING_SMOOTHING)
                last_average_palm_width += (raw_palm_width - last_average_palm_width) * TRACKING_SMOOTHING
                last_hand_distance += (hand_distance - last_hand_distance) * TRACKING_SMOOTHING

            forming = True

    if forming:
        lost_detection_timer = 0.0
    else:
        lost_detection_timer += dt

    effective_forming = forming or (lost_detection_timer < FORMING_GRACE_PERIOD)

    if effective_forming:
        charge_time = min(charge_time + dt, CHARGE_DURATION)
    else:
        charge_time = max(charge_time - dt * CHARGE_DRAIN_RATE, 0.0)

    charge_fraction = charge_time / CHARGE_DURATION

    if charge_fraction > 0.02:
        mid_x = last_mid_x
        mid_y = last_mid_y

        target_size = MIN_SIZE + (MAX_SIZE - MIN_SIZE) * charge_fraction
        target_size = target_size * np.clip(last_average_palm_width / 90.0, 0.7, 1.4)

        # Limit the sphere size to the gap between the palms.
        target_size = min(target_size, last_hand_distance * 1.3)

        size = int(np.clip(target_size, MIN_SIZE, MAX_SIZE))

        jitter = float(rng.normal(0, 0.03 + (1 - charge_fraction) * 0.05))
        size = int(size * (1 + rng.normal(0, 0.01)))

        t = now - start_time
        texture = build_rasengan_texture(size, t, charge_fraction, jitter)

        x = mid_x - size // 2
        y = mid_y - size // 2

        frame = composite_rasengan(frame, x, y, texture)

    cv2.imshow("Rasengan", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
