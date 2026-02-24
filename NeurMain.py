import cv2
import pickle
import numpy as np
import os
import uuid
import platform
import shutil
import warnings
import time

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

import face_recognition

# Абсолютный путь к папке проекта
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

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


def log_visibility_event(name, face_id, event_text, duration_sec):
    common_log = os.path.join(PROJECT_DIR, "log.txt")
    per_id_log = os.path.join(LOGS_DIR, f"{face_id}.txt")
    log_lines = [
        f"{event_text}: {name} (ID: {face_id})",
        f"Время в зоне видимости: {duration_sec:.1f} сек",
        "-" * 80,
    ]
    log_text = "\n".join(log_lines) + "\n"

    with open(common_log, "a", encoding="utf-8") as f:
        f.write(log_text)
    with open(per_id_log, "a", encoding="utf-8") as f:
        f.write(log_text)
    print(f"[LOG] {event_text}: ID={face_id}, name={name}, visible={duration_sec:.1f}s")


def format_duration(seconds):
    total = int(max(0, seconds))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"




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
ENCODING_ADD_COOLDOWN_FRAMES = 10
DETECTION_FRAME_INTERVAL = 8
DETECTION_SCALE = 0.20
MAX_FACES_PER_FRAME = 2
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
CAPTURE_FPS = 30


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
# Единые настройки потока: нормальная картинка для вывода и меньше лагов.
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, CAPTURE_FPS)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

frame_count = 0
active_faces = []
recent_tracks = {}  # user_id -> {"bbox": (t, r, b, l), "last_seen_frame": int}
last_encoding_add_frame = {}
visibility_stats = {}  # user_id -> {"name": str, "total_sec": float, "visible_since": float|None, "last_seen_ts": float}

print("[INFO] Нажми 'q' для выхода, 'w' — редактировать имя, 'd' — удалить лицо.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Ошибка чтения кадра")
        break


    frame = cv2.flip(frame, 1)
    display_frame = frame.copy()

    if frame_count % DETECTION_FRAME_INTERVAL == 0:
        # Тяжёлую детекцию и сопоставление делаем раз в N кадров.
        small_frame = cv2.resize(frame, (0, 0), fx=DETECTION_SCALE, fy=DETECTION_SCALE)
        # dlib ожидает contiguous-массив; срез ::-1 даёт view с отрицательным stride.
        rgb_small_frame = small_frame[:, :, ::-1].copy()
        face_locations_small = face_recognition.face_locations(rgb_small_frame, model="hog")
        if len(face_locations_small) > MAX_FACES_PER_FRAME:
            face_locations_small = sorted(
                face_locations_small,
                key=lambda box: (box[2] - box[0]) * (box[1] - box[3]),
                reverse=True,
            )[:MAX_FACES_PER_FRAME]
        face_encodings_small = face_recognition.face_encodings(
            rgb_small_frame,
            face_locations_small,
            num_jitters=1,
            model="small",
        )

        active_faces = []
        seen_known_ids_this_frame = set()

        for face_encoding, (top, right, bottom, left) in zip(face_encodings_small, face_locations_small):
            # Масштабируем координаты обратно под исходный кадр.
            scale_inv = 1.0 / DETECTION_SCALE
            top = int(top * scale_inv)
            right = int(right * scale_inv)
            bottom = int(bottom * scale_inv)
            left = int(left * scale_inv)
            top = max(0, top)
            left = max(0, left)
            right = min(frame.shape[1], right)
            bottom = min(frame.shape[0], bottom)

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
                    matched_id = new_id

            if matched_id is not None:
                seen_known_ids_this_frame.add(matched_id)
                recent_tracks[matched_id] = {"bbox": current_box, "last_seen_frame": frame_count}

            active_faces.append(
                {
                    "box": current_box,
                    "name": name,
                    "id": matched_id,
                }
            )

    frame_count += 1

    now_ts = time.time()
    currently_visible_ids = {face_item["id"] for face_item in active_faces if face_item.get("id")}
    for face_item in active_faces:
        user_id = face_item.get("id")
        if not user_id:
            continue
        stats = visibility_stats.setdefault(
            user_id,
            {
                "name": face_item["name"],
                "total_sec": 0.0,
                "visible_since": None,
                "last_seen_ts": 0.0,
            },
        )
        stats["name"] = face_item["name"]
        if stats["visible_since"] is None:
            stats["visible_since"] = now_ts
        stats["last_seen_ts"] = now_ts

    for user_id, stats in visibility_stats.items():
        if stats["visible_since"] is None:
            continue
        if user_id not in currently_visible_ids:
            session_sec = now_ts - stats["visible_since"]
            stats["total_sec"] += max(0.0, session_sec)
            stats["visible_since"] = None

    # Между тяжёлыми кадрами только рисуем уже готовые результаты.
    for face_item in active_faces:
        top, right, bottom, left = face_item["box"]
        cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
        user_id = face_item.get("id")
        visible_text = "00:00"
        if user_id and user_id in visibility_stats:
            stats = visibility_stats[user_id]
            elapsed_sec = stats["total_sec"]
            if stats["visible_since"] is not None:
                elapsed_sec += now_ts - stats["visible_since"]
            visible_text = format_duration(elapsed_sec)
        cv2.putText(display_frame, face_item["name"], (left, max(20, top - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(display_frame, f"time {visible_text}", (left, bottom + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

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
            if selected in visibility_stats:
                del visibility_stats[selected]

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
final_ts = time.time()
for user_id, stats in visibility_stats.items():
    if stats["visible_since"] is not None:
        stats["total_sec"] += max(0.0, final_ts - stats["visible_since"])
        stats["visible_since"] = None
    log_visibility_event(stats["name"], user_id, "Итог по зоне видимости", stats["total_sec"])
    print(f"[INFO] ID {user_id} был в зоне видимости {format_duration(stats['total_sec'])}")
print("[INFO] Завершение. База сохранена.")
