"""
BaseMemeCV - Real-time Face Expression Meme Viewer

An OpenCV + MediaPipe app that detects facial expressions from webcam input
and displays corresponding Base meme images.
"""

import os

import cv2
import mediapipe as mp

# Initialize MediaPipe FaceMesh
face_mesh = mp.solutions.face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_faces=1,
)

# Detection thresholds (kept from original baseline)
eye_opening_threshold = 0.025
mouth_open_threshold = 0.03
squinting_threshold = 0.018

# Expression -> meme image mapping
ASSET_MAP = {
    "shock": "assets/brett-shock.png",
    "tongue": "assets/toshi-tongue.png",
    "glare": "assets/brett-glare.png",
    "fallback": "assets/larry-base.png",
}


def is_shock(face_landmarks):
    """Detect 'shock' by checking if eyes are widely open."""
    l_top = face_landmarks.landmark[159]
    l_bot = face_landmarks.landmark[145]
    r_top = face_landmarks.landmark[386]
    r_bot = face_landmarks.landmark[374]

    eye_opening = (abs(l_top.y - l_bot.y) + abs(r_top.y - r_bot.y)) / 2.0
    return eye_opening > eye_opening_threshold


def is_tongue(face_landmarks):
    """Detect 'tongue' style expression by checking mouth opening."""
    top_lip = face_landmarks.landmark[13]
    bottom_lip = face_landmarks.landmark[14]

    mouth_open = abs(top_lip.y - bottom_lip.y)
    return mouth_open > mouth_open_threshold


def is_glare(face_landmarks):
    """Detect 'glare' by checking if eyes are squinting."""
    l_top = face_landmarks.landmark[159]
    l_bot = face_landmarks.landmark[145]
    r_top = face_landmarks.landmark[386]
    r_bot = face_landmarks.landmark[374]

    eye_squint = (abs(l_top.y - l_bot.y) + abs(r_top.y - r_bot.y)) / 2.0
    return eye_squint < squinting_threshold


def select_meme(face_landmarks):
    """Select meme image by priority: tongue > shock > glare > fallback."""
    if is_tongue(face_landmarks):
        return ASSET_MAP["tongue"]
    if is_shock(face_landmarks):
        return ASSET_MAP["shock"]
    if is_glare(face_landmarks):
        return ASSET_MAP["glare"]
    return ASSET_MAP["fallback"]


def draw_landmarks(frame, face_landmarks):
    """Draw face landmarks for debugging."""
    height, width = frame.shape[:2]
    for lm in face_landmarks.landmark:
        x = int(lm.x * width)
        y = int(lm.y * height)
        cv2.circle(frame, (x, y), 1, (0, 180, 0), -1)


def show_meme_window(base_frame, image_path):
    """Render meme image window or show a missing-file warning."""
    meme = cv2.imread(image_path)
    if meme is not None:
        meme = cv2.resize(meme, (640, 480))
        cv2.imshow("Base Meme Image", meme)
        return

    warning = base_frame * 0
    cv2.putText(
        warning,
        f"Missing file: {image_path}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
    )
    cv2.imshow("Base Meme Image", warning)


def validate_assets():
    """Print warnings for missing asset files before app starts."""
    missing = [path for path in ASSET_MAP.values() if not os.path.exists(path)]
    if missing:
        print("[WARN] Missing asset files:")
        for path in missing:
            print(f"  - {path}")


def main():
    validate_assets()

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("[ERROR] Cannot open webcam (device 0).")
        print("Please check camera permission or close other apps using webcam.")
        return

    while True:
        ret, frame = cam.read()
        if not ret:
            print("[ERROR] Failed to read frame from webcam.")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed = face_mesh.process(rgb_frame)
        faces = processed.multi_face_landmarks

        meme_image = ASSET_MAP["shock"]  # default

        if faces:
            face_landmarks = faces[0]
            meme_image = select_meme(face_landmarks)
            draw_landmarks(frame, face_landmarks)

        cv2.imshow("Face Detection", frame)
        show_meme_window(frame, meme_image)

        key = cv2.waitKey(1)
        if key == 27:  # ESC to quit
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
