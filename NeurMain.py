import cv2
import pickle
import numpy as np
import os
import dlib
import uuid
import platform
import shutil
import warnings

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

import face_recognition

# Абсолютный путь к папке проекта
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Папка проекта и моделей

MODELS_DIR = os.path.join(PROJECT_DIR, "models")


predictor_path = os.path.join(MODELS_DIR, "shape_predictor_68_face_landmarks.dat")
rec_model_path = os.path.join(MODELS_DIR, "dlib_face_recognition_resnet_model_v1.dat")

predictor = dlib.shape_predictor(predictor_path) # pyright: ignore[reportAttributeAccessIssue]
face_rec_model = dlib.face_recognition_model_v1(rec_model_path) # pyright: ignore[reportAttributeAccessIssue]

# Папка с базой и фотографиями
BASE_FILE = os.path.join(PROJECT_DIR, "known_faces.pkl")
FACES_DIR = os.path.join(PROJECT_DIR, "faces")
LOGS_DIR = os.path.join(PROJECT_DIR, "logs")


def log_face_event(name, face_id, encoding, event_text, total_faces):
    common_log = os.path.join(PROJECT_DIR, "log.txt")
    per_id_log = os.path.join(LOGS_DIR, f"{face_id}.txt")
    log_lines = [
        f"{event_text}: {name} (ID: {face_id})",
        "Кодировка:",
        f"{encoding}",
        f"Всего ID в базе: {total_faces}",
        "-" * 80,
    ]
    log_text = "\n".join(log_lines) + "\n"

    with open(common_log, "a", encoding="utf-8") as f:
        f.write(log_text)
    with open(per_id_log, "a", encoding="utf-8") as f:
        f.write(log_text)
    print(f"[LOG] {event_text}: ID={face_id}, name={name}, total_ids={total_faces}")




# Создание папки для фотографий
if not os.path.exists(FACES_DIR):
    os.makedirs(FACES_DIR)
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)

# Загрузка базы
if os.path.exists(BASE_FILE):
    with open(BASE_FILE, "rb") as f:
        known_faces = pickle.load(f)
    print(f"[INFO] Загружено {len(known_faces)} ID.")
else:
    known_faces = {}
    print("[INFO] База не найдена. Создаём новую.")

# Утилита
def build_encodings_dict(face_dict):
    encodings, ids, names = [], [], []
    for user_id, data in face_dict.items():
        for enc in data["encodings"]:
            encodings.append(enc)
            ids.append(user_id)
            names.append(data["name"])
    return encodings, ids, names

known_encodings, known_ids, known_names = build_encodings_dict(known_faces)

# Параметры сопоставления и накопления данных
FACE_MATCH_THRESHOLD = 0.6
TRACK_MAX_MISSING_FRAMES = 30
POSITION_IOU_THRESHOLD = 0.20
POSITION_CENTER_RATIO_THRESHOLD = 0.45
ENCODING_ADD_MIN_DISTANCE = 0.035
ENCODING_ADD_COOLDOWN_FRAMES = 20


def save_base():
    with open(BASE_FILE, "wb") as f:
        pickle.dump(known_faces, f)


def get_face_dir(user_id):
    face_dir = os.path.join(FACES_DIR, user_id)
    os.makedirs(face_dir, exist_ok=True)
    return face_dir


def bbox_iou(box_a, box_b):
    a_top, a_right, a_bottom, a_left = box_a
    b_top, b_right, b_bottom, b_left = box_b

    inter_left = max(a_left, b_left)
    inter_top = max(a_top, b_top)
    inter_right = min(a_right, b_right)
    inter_bottom = min(a_bottom, b_bottom)

    inter_w = max(0, inter_right - inter_left)
    inter_h = max(0, inter_bottom - inter_top)
    inter_area = inter_w * inter_h

    area_a = max(0, a_right - a_left) * max(0, a_bottom - a_top)
    area_b = max(0, b_right - b_left) * max(0, b_bottom - b_top)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def center_distance_ratio(box_a, box_b):
    a_top, a_right, a_bottom, a_left = box_a
    b_top, b_right, b_bottom, b_left = box_b

    ax = (a_left + a_right) / 2.0
    ay = (a_top + a_bottom) / 2.0
    bx = (b_left + b_right) / 2.0
    by = (b_top + b_bottom) / 2.0

    distance = np.sqrt((ax - bx) ** 2 + (ay - by) ** 2)
    norm = max(
        1.0,
        max(a_right - a_left, a_bottom - a_top, b_right - b_left, b_bottom - b_top),
    )
    return distance / norm


def find_recent_id_by_position(current_box, current_frame, tracks, used_ids):
    best_id = None
    best_score = -1.0

    for user_id, info in tracks.items():
        if user_id in used_ids or user_id not in known_faces:
            continue

        if current_frame - info["last_seen_frame"] > TRACK_MAX_MISSING_FRAMES:
            continue

        prev_box = info["bbox"]
        iou = bbox_iou(current_box, prev_box)
        center_ratio = center_distance_ratio(current_box, prev_box)

        if iou >= POSITION_IOU_THRESHOLD or center_ratio <= POSITION_CENTER_RATIO_THRESHOLD:
            score = iou - center_ratio * 0.1
            if score > best_score:
                best_score = score
                best_id = user_id

    return best_id


def save_face_image(user_id, face_img, primary=False):
    if face_img.size == 0:
        return None
    face_dir = get_face_dir(user_id)
    if primary:
        filename = "main.jpg"
    else:
        filename = f"{uuid.uuid4().hex[:8]}.jpg"
    face_path = os.path.join(face_dir, filename)
    cv2.imwrite(face_path, face_img)
    return face_path


def try_add_encoding_to_existing(user_id, face_encoding, face_img, current_frame, last_add_frame):
    enc_list = known_faces[user_id]["encodings"]
    min_distance = 1.0

    if enc_list:
        min_distance = float(np.min(face_recognition.face_distance(enc_list, face_encoding)))

    frame_gap = current_frame - last_add_frame.get(user_id, -10**9)
    if min_distance <= ENCODING_ADD_MIN_DISTANCE or frame_gap < ENCODING_ADD_COOLDOWN_FRAMES:
        return False, None

    known_faces[user_id]["encodings"].append(face_encoding)
    face_path = save_face_image(user_id, face_img, primary=False)
    last_add_frame[user_id] = current_frame
    return True, face_path


# Камера
def open_camera():
    system = platform.system().lower()
    candidates = []

    if system == "windows":
        # DirectShow only on Windows
        candidates = [(0, cv2.CAP_DSHOW), (0, cv2.CAP_MSMF), (1, cv2.CAP_DSHOW)]
    else:
        # Linux/macOS: default backend, plus V4L2 when available
        candidates = [(0, None), (1, None)]
        if hasattr(cv2, "CAP_V4L2"):
            candidates.extend([(0, cv2.CAP_V4L2), (1, cv2.CAP_V4L2)])  # pyright: ignore[reportArgumentType]

    for index, backend in candidates:
        cap_try = cv2.VideoCapture(index) if backend is None else cv2.VideoCapture(index, backend)
        if cap_try.isOpened():
            print(f"[INFO] Камера открыта: index={index}, backend={backend}")
            return cap_try
        cap_try.release()

    return None


cap = open_camera()
if cap is None or not cap.isOpened():
    print("Не удалось открыть камеру (проверьте права доступа к /dev/video* и занятость устройства)")
    exit()

frame_count = 0
scaled_locations, scaled_encodings = [], []
recent_tracks = {}  # user_id -> {"bbox": (t, r, b, l), "last_seen_frame": int}
last_encoding_add_frame = {}

print("[INFO] Нажми 'q' для выхода, 'w' — редактировать имя, 'd' — удалить лицо.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Ошибка чтения кадра")
        break


    frame = cv2.flip(frame, 1)
    display_frame = frame.copy()

    if frame_count % 10 == 0:
        # Уменьшаем
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")

        scaled_locations, scaled_encodings = [], []

        for top, right, bottom, left in face_locations:
            # масштаб
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            scaled_locations.append((top, right, bottom, left))

            rect = dlib.rectangle(left, top, right, bottom) # pyright: ignore[reportAttributeAccessIssue]
            shape = predictor(frame, rect)
            descriptor = face_rec_model.compute_face_descriptor(frame, shape)
            face_encoding = np.array(descriptor)
            scaled_encodings.append(face_encoding)

    frame_count += 1

    seen_known_ids_this_frame = set()

    for face_encoding, (top, right, bottom, left) in zip(scaled_encodings, scaled_locations):
        name = "Unknown"
        matched_id = None
        current_box = (top, right, bottom, left)
        face_img = frame[top:bottom, left:right]

        if known_encodings:
            distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_index = np.argmin(distances)

            if distances[best_index] < FACE_MATCH_THRESHOLD:
                matched_id = known_ids[best_index]
                if matched_id in known_faces:
                    name = known_faces[matched_id]["name"]

        if matched_id is None:
            # Если лицо только что пропадало и снова появилось рядом, считаем что это тот же ID.
            fallback_id = find_recent_id_by_position(
                current_box,
                frame_count,
                recent_tracks,
                seen_known_ids_this_frame,
            )
            if fallback_id is not None:
                matched_id = fallback_id
                name = known_faces[fallback_id]["name"]
                added, added_path = try_add_encoding_to_existing(
                    fallback_id,
                    face_encoding,
                    face_img,
                    frame_count,
                    last_encoding_add_frame,
                )
                if added:
                    save_base()
                    log_face_event(
                        known_faces[fallback_id]["name"],
                        fallback_id,
                        face_encoding,
                        "Дополнен ID",
                        len(known_faces),
                    )
                    known_encodings, known_ids, known_names = build_encodings_dict(known_faces)
                    print(f"[INFO] Дополнен ID {fallback_id}: +encoding, фото {added_path}")
            else:
                print("\n=== ДОБАВЛЕНИЕ НОВОГО ЛИЦА ===")
                new_id = str(uuid.uuid4())
                known_faces[new_id] = {
                    "name": "Unknown",
                    "encodings": [face_encoding]
                }
                face_path = save_face_image(new_id, face_img, primary=True)
                save_base()
                log_face_event("Unknown", new_id, face_encoding, "Добавлено лицо", len(known_faces))
                known_encodings, known_ids, known_names = build_encodings_dict(known_faces)
                print(f"[INFO] Добавлено новое лицо: {new_id}, фото сохранено как {face_path}")

        if matched_id is not None:
            seen_known_ids_this_frame.add(matched_id)
            recent_tracks[matched_id] = {"bbox": current_box, "last_seen_frame": frame_count}

        # Рисуем лицо
        cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(display_frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Очищаем слишком старые треки
    stale_ids = [
        user_id for user_id, info in recent_tracks.items()
        if frame_count - info["last_seen_frame"] > TRACK_MAX_MISSING_FRAMES
    ]
    for user_id in stale_ids:
        del recent_tracks[user_id]

    # Отображаем
    cv2.imshow("Camera", display_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('w'):
        print("\n=== РЕДАКТИРОВАНИЕ ИМЕНИ ===")
        for uid in known_faces:
            print(f"{uid}: {known_faces[uid]['name']}")
        selected = input("Введите ID для переименования: ").strip()
        if selected in known_faces:
            new_name = input("Введите новое имя: ").strip()
            known_faces[selected]["name"] = new_name
            save_base()
            
            print(f"[INFO] Имя для {selected} обновлено на {new_name}")
        else:
            print("[WARN] Неверный ID")

    elif key == ord('d'):
        print("\n=== УДАЛЕНИЕ ЛИЦА ===")
        for uid in known_faces:
            print(f"{uid}: {known_faces[uid]['name']}")
        selected = input("Введите ID для удаления: ").strip()
        if selected in known_faces:
            # Удаляем лицо из базы
            del known_faces[selected]

            # Удаляем папку с фото ID
            face_dir = os.path.join(FACES_DIR, selected)
            if os.path.isdir(face_dir):
                shutil.rmtree(face_dir)
                print(f"[INFO] Папка фото удалена: {face_dir}")

            # Удаляем отдельный лог ID
            per_id_log = os.path.join(LOGS_DIR, f"{selected}.txt")
            if os.path.exists(per_id_log):
                os.remove(per_id_log)
                print(f"[INFO] Лог ID удалён: {per_id_log}")

            save_base()
            print(f"[INFO] Лицо с ID {selected} удалено.")

            # Перестроим вспомогательные списки
            known_encodings, known_ids, known_names = build_encodings_dict(known_faces)
        else:
            print("[WARN] Неверный ID")

cap.release()
cv2.destroyAllWindows()
print("[INFO] Завершение. База сохранена.")
