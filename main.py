"""
BaseMemeCV - Base Chain Meme Edition

A openCV + MediaPipe program that detects facial expressions 
and displays BASE CHAIN memes (Brett, Toshi, Degen...) in real time.
"""

import cv2
import mediapipe as mp

# ====================== KHỞI TẠO ======================
face_mesh = mp.solutions.face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_faces=1
)

cam = cv2.VideoCapture(0)

# ====================== THRESHOLDS (có thể chỉnh) ======================
eye_opening_threshold = 0.025   # Shock (mắt trợn to)
mouth_open_threshold = 0.03     # Tongue (miệng thè lưỡi)
squinting_threshold = 0.018     # Glare (mắt lườm/híp)

# ====================== HÀM PHÁT HIỆN BIỂU CẢM ======================
def meme_shock(face_landmark_points):
    l_top = face_landmark_points.landmark[159]
    l_bot = face_landmark_points.landmark[145]
    r_top = face_landmark_points.landmark[386]
    r_bot = face_landmark_points.landmark[374]

    eye_opening = (abs(l_top.y - l_bot.y) + abs(r_top.y - r_bot.y)) / 2.0
    return eye_opening > eye_opening_threshold

def meme_tongue(face_landmark_points):
    top_lip = face_landmark_points.landmark[13]
    bottom_lip = face_landmark_points.landmark[14]

    mouth_open = abs(top_lip.y - bottom_lip.y)
    return mouth_open > mouth_open_threshold

def meme_glare(face_landmark_points):
    l_top = face_landmark_points.landmark[159]
    l_bot = face_landmark_points.landmark[145]
    r_top = face_landmark_points.landmark[386]
    r_bot = face_landmark_points.landmark[374]

    eye_squint = (abs(l_top.y - l_bot.y) + abs(r_top.y - r_bot.y)) / 2.0
    return eye_squint < squinting_threshold

# ====================== MAIN ======================
def main():
    while True:
        ret, image = cam.read()
        if not ret:
            break

        image = cv2.flip(image, 1)                    # Lật ngang (mirror)
        height, width, _ = image.shape

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        processed_image = face_mesh.process(rgb_image)
        face_landmark_points = processed_image.multi_face_landmarks

        # Mặc định là meme Toshi chill
        meme_image_path = "assets/toshi-neutral.png"

        if face_landmark_points:
            face_landmark_points = face_landmark_points[0]

            # Ưu tiên: Tongue > Shock > Glare
            if meme_tongue(face_landmark_points):
                meme_image_path = "assets/toshi-tongue.png"
            elif meme_shock(face_landmark_points):
                meme_image_path = "assets/brett-shock.png"
            elif meme_glare(face_landmark_points):
                meme_image_path = "assets/brett-glare.png"
            else:
                meme_image_path = "assets/toshi-neutral.png"

            # Vẽ landmarks lên mặt (xanh lá nhỏ)
            for lm in face_landmark_points.landmark:
                x = int(lm.x * width)
                y = int(lm.y * height)
                cv2.circle(image, (x, y), 1, (0, 100, 0), -1)

        cv2.imshow('BaseMemeCV - Webcam', image)

        # ==================== HIỂN THỊ MEME ====================
        meme = cv2.imread(meme_image_path, cv2.IMREAD_UNCHANGED)  # Hỗ trợ PNG transparent

        if meme is not None:
            # Resize về kích thước đẹp (có thể chỉnh)
            meme = cv2.resize(meme, (640, 480))
            cv2.imshow("Base Meme Reaction", meme)
        else:
            # Nếu thiếu file ảnh
            blank = image * 0
            cv2.putText(blank, f"Missing: {meme_image_path}", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Base Meme Reaction", blank)

        key = cv2.waitKey(1)
        if key == 27:   # ESC để thoát
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("🚀 BaseMemeCV đang chạy... Nhấn ESC để thoát!")
    main()
