import cv2
import pickle
import numpy as np
import os
import uuid
import platform
import shutil
import warnings
import time
import multiprocessing as mp
import queue

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

import face_recognition

# Absolute path to the project folder
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Folder with database and photos
BASE_FILE = os.path.join(PROJECT_DIR, "known_faces.pkl")
FACES_DIR = os.path.join(PROJECT_DIR, "faces")
LOGS_DIR = os.path.join(PROJECT_DIR, "logs")


def log_face_event(name, face_id, encoding, event_text, total_faces):
    common_log = os.path.join(PROJECT_DIR, "log.txt")
    per_id_log = os.path.join(LOGS_DIR, f"{face_id}.txt")
    log_lines = [
        f"{event_text}: {name} (ID: {face_id})",
        "Encoding:",
        f"{encoding}",
        f"Total IDs in database: {total_faces}",
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
        f"Time in visibility zone: {duration_sec:.1f} sec",
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


if not os.path.exists(FACES_DIR):
    os.makedirs(FACES_DIR)
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)

if os.path.exists(BASE_FILE):
    with open(BASE_FILE, "rb") as f:
        known_faces = pickle.load(f)
    print(f"[INFO] Loaded {len(known_faces)} IDs.")
else:
    known_faces = {}
    print("[INFO] Database not found. Creating new one.")


def build_encodings_dict(face_dict):
    encodings, ids, names = [], [], []
    for user_id, data in face_dict.items():
        for enc in data["encodings"]:
            encodings.append(enc)
            ids.append(user_id)
            names.append(data["name"])
    return encodings, ids, names


def get_next_unknown_name():
    max_unknown_number = 0
    for data in known_faces.values():
        stored_name = data.get("name", "")
        if stored_name == "Unknown":
            max_unknown_number = max(max_unknown_number, 1)
            continue

        if not stored_name.startswith("Unknown "):
            continue

        suffix = stored_name[len("Unknown ") :].strip()
        if suffix.isdigit():
            max_unknown_number = max(max_unknown_number, int(suffix))

    return f"Unknown {max_unknown_number + 1 if max_unknown_number else 1}"


known_encodings, known_ids, known_names = build_encodings_dict(known_faces)

# Matching and data accumulation parameters
FACE_MATCH_THRESHOLD = 0.6
TRACK_MAX_MISSING_FRAMES = 30
POSITION_IOU_THRESHOLD = 0.20
POSITION_CENTER_RATIO_THRESHOLD = 0.45
ENCODING_ADD_MIN_DISTANCE = 0.035
ENCODING_ADD_COOLDOWN_FRAMES = 10
# Added compared with main.py:
# this ceiling-distance profile spends more CPU on larger faces and denser scans.
DETECTION_FRAME_INTERVAL = 2
DETECTION_SCALE = 0.45
MAX_FACES_PER_FRAME = 6
# Added compared with main.py:
# keep visibility active through longer recognition dropouts from a difficult viewpoint.
VISIBILITY_LOST_TIMEOUT_SEC = 3.0
# Added compared with main.py:
# retain the last known face position for longer when a detection cycle misses.
TRACK_HOLD_FRAMES = 20
# Added compared with main.py:
# scan a few in-plane rotations and upsample more aggressively for small distant faces.
DETECTION_UPSAMPLE = 1
DETECTION_ANGLES = (-18, 0, 18)
ROTATION_DUPLICATE_IOU = 0.35
# Added compared with main.py:
# request a larger capture size so distant faces occupy more pixels before downscaling.
CAPTURE_WIDTH = 1920
CAPTURE_HEIGHT = 1080
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
    filename = "main.jpg" if primary else f"{uuid.uuid4().hex[:8]}.jpg"
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


def open_camera():
    system = platform.system().lower()
    candidates = []

    if system == "windows":
        candidates = [(0, cv2.CAP_DSHOW), (0, cv2.CAP_MSMF), (1, cv2.CAP_DSHOW)]
    else:
        candidates = [(0, None), (1, None)]
        if hasattr(cv2, "CAP_V4L2"):
            candidates.extend([(0, cv2.CAP_V4L2), (1, cv2.CAP_V4L2)])

    for index, backend in candidates:
        cap_try = cv2.VideoCapture(index) if backend is None else cv2.VideoCapture(index, backend)
        if cap_try.isOpened():
            print(f"[INFO] Camera opened: index={index}, backend={backend}")
            return cap_try
        cap_try.release()

    return None


def preprocess_small_frame(frame):
    # Added compared with main.py:
    # boost contrast and edge detail so small distant faces survive downscaling better.
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
    enhanced = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.2)
    return cv2.addWeighted(enhanced, 1.35, blurred, -0.35, 0)


def rotate_small_frame(frame, angle):
    height, width = frame.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        frame,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated, matrix


def map_box_to_original(box, inverse_matrix, width, height):
    top, right, bottom, left = box
    corners = np.array(
        [
            [[left, top]],
            [[right, top]],
            [[right, bottom]],
            [[left, bottom]],
        ],
        dtype=np.float32,
    )
    mapped = cv2.transform(corners, inverse_matrix).reshape(-1, 2)
    xs = mapped[:, 0]
    ys = mapped[:, 1]

    mapped_left = max(0, int(np.floor(np.min(xs))))
    mapped_top = max(0, int(np.floor(np.min(ys))))
    mapped_right = min(width, int(np.ceil(np.max(xs))))
    mapped_bottom = min(height, int(np.ceil(np.max(ys))))

    if mapped_right <= mapped_left or mapped_bottom <= mapped_top:
        return None
    return (mapped_top, mapped_right, mapped_bottom, mapped_left)


def detection_worker(task_queue, result_queue):
    # Difference from low_main.py:
    # face detection/encoding runs in a separate process instead of the main UI loop.
    # Added compared with main.py:
    # this ceiling variant enhances contrast, requests more detail, and scans rotations.
    while True:
        task = task_queue.get()
        if task is None:
            break

        frame_id = task["frame_id"]
        small_frame = preprocess_small_frame(task["small_frame"])
        height, width = small_frame.shape[:2]
        detections = []

        for angle in DETECTION_ANGLES:
            rotated_frame, matrix = rotate_small_frame(small_frame, angle)
            inverse_matrix = cv2.invertAffineTransform(matrix)
            rgb_rotated_frame = rotated_frame[:, :, ::-1].copy()
            rotated_locations = face_recognition.face_locations(
                rgb_rotated_frame,
                number_of_times_to_upsample=DETECTION_UPSAMPLE,
                model="hog",
            )
            if not rotated_locations:
                continue

            rotated_encodings = face_recognition.face_encodings(
                rgb_rotated_frame,
                rotated_locations,
                num_jitters=1,
                model="small",
            )

            for rotated_box, face_encoding in zip(rotated_locations, rotated_encodings):
                mapped_box = map_box_to_original(rotated_box, inverse_matrix, width, height)
                if mapped_box is None:
                    continue

                duplicate = False
                for existing in detections:
                    if bbox_iou(existing["box"], mapped_box) >= ROTATION_DUPLICATE_IOU:
                        duplicate = True
                        break
                if duplicate:
                    continue

                detections.append(
                    {
                        "box": mapped_box,
                        "encoding": face_encoding,
                        "area": (mapped_box[2] - mapped_box[0]) * (mapped_box[1] - mapped_box[3]),
                    }
                )

        detections.sort(key=lambda item: item["area"], reverse=True)
        detections = detections[:MAX_FACES_PER_FRAME]
        face_locations_small = [item["box"] for item in detections]
        face_encodings_small = [item["encoding"] for item in detections]

        result_queue.put(
            {
                "frame_id": frame_id,
                "face_locations_small": face_locations_small,
                "face_encodings_small": face_encodings_small,
            }
        )


def drain_latest_result(result_queue):
    # Difference from low_main.py:
    # the main process polls completed worker results without blocking frame rendering.
    latest = None
    while True:
        try:
            latest = result_queue.get_nowait()
        except queue.Empty:
            return latest


def build_held_faces(frame_count, recent_tracks):
    held_faces = []
    for user_id, info in recent_tracks.items():
        if frame_count - info["last_seen_frame"] > TRACK_HOLD_FRAMES:
            continue
        if user_id not in known_faces:
            continue
        held_faces.append(
            {
                "box": info["bbox"],
                "name": known_faces[user_id]["name"],
                "id": user_id,
                "held": True,
            }
        )
    return held_faces


def process_detection_result(result, frame, frame_count, active_faces, recent_tracks, last_encoding_add_frame):
    global known_encodings, known_ids, known_names

    face_locations_small = result["face_locations_small"]
    face_encodings_small = result["face_encodings_small"]
    updated_faces = []
    seen_known_ids_this_frame = set()

    for face_encoding, (top, right, bottom, left) in zip(face_encodings_small, face_locations_small):
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
            best_index = int(np.argmin(distances))

            if distances[best_index] < FACE_MATCH_THRESHOLD:
                matched_id = known_ids[best_index]
                if matched_id in known_faces:
                    name = known_faces[matched_id]["name"]

        if matched_id is None:
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
                        "ID Supplemented",
                        len(known_faces),
                    )
                    known_encodings, known_ids, known_names = build_encodings_dict(known_faces)
                    print(f"[INFO] ID {fallback_id} supplemented: +encoding, photo {added_path}")
            else:
                print("\n=== ADDING NEW FACE ===")
                new_id = str(uuid.uuid4())
                name = get_next_unknown_name()
                known_faces[new_id] = {"name": name, "encodings": [face_encoding]}
                face_path = save_face_image(new_id, face_img, primary=True)
                save_base()
                log_face_event(name, new_id, face_encoding, "Face Added", len(known_faces))
                known_encodings, known_ids, known_names = build_encodings_dict(known_faces)
                print(f"[INFO] New face added: {new_id}, photo saved as {face_path}")
                matched_id = new_id

        if matched_id is not None:
            seen_known_ids_this_frame.add(matched_id)
            recent_tracks[matched_id] = {"bbox": current_box, "last_seen_frame": frame_count}

        updated_faces.append({"box": current_box, "name": name, "id": matched_id, "held": False})

    if updated_faces:
        return updated_faces
    return build_held_faces(frame_count, recent_tracks)


def main():
    global known_encodings, known_ids, known_names

    # Difference from low_main.py:
    # this variant explicitly prepares the main process for a multi-process pipeline.
    cv2.setNumThreads(os.cpu_count() or 1)

    cap = open_camera()
    if cap is None or not cap.isOpened():
        print("Failed to open camera (check access rights to /dev/video* and device availability)")
        return

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAPTURE_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Difference from low_main.py:
    # queues and a dedicated worker process replace in-loop synchronous detection.
    task_queue = mp.Queue(maxsize=1)
    result_queue = mp.Queue(maxsize=1)
    worker = mp.Process(target=detection_worker, args=(task_queue, result_queue), daemon=True)
    worker.start()

    frame_count = 0
    active_faces = []
    recent_tracks = {}
    last_encoding_add_frame = {}
    visibility_stats = {}
    frame_cache = {}
    pending_frame_id = None

    print("[INFO] Press 'q' to exit, 'w' - edit name, 'd' - delete face.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame reading error")
                break

            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()

            latest_result = drain_latest_result(result_queue)
            if latest_result is not None:
                processed_frame = frame_cache.pop(latest_result["frame_id"], None)
                if processed_frame is not None:
                    # Difference from low_main.py:
                    # recognition results arrive later and are applied when the worker finishes.
                    active_faces = process_detection_result(
                        latest_result,
                        processed_frame,
                        latest_result["frame_id"],
                        active_faces,
                        recent_tracks,
                        last_encoding_add_frame,
                    )
                pending_frame_id = None
                frame_cache.clear()
            elif active_faces and all(face_item.get("held") for face_item in active_faces):
                active_faces = build_held_faces(frame_count, recent_tracks)

            if frame_count % DETECTION_FRAME_INTERVAL == 0 and pending_frame_id is None:
                small_frame = cv2.resize(frame, (0, 0), fx=DETECTION_SCALE, fy=DETECTION_SCALE)
                payload = {"frame_id": frame_count, "small_frame": small_frame}
                try:
                    # Difference from low_main.py:
                    # only the reduced frame is handed to the worker, while the UI loop keeps running.
                    task_queue.put_nowait(payload)
                    frame_cache[frame_count] = frame.copy()
                    pending_frame_id = frame_count
                except queue.Full:
                    pass

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
                if not face_item.get("held"):
                    stats["last_seen_ts"] = now_ts

            for user_id, stats in visibility_stats.items():
                if stats["visible_since"] is None:
                    continue
                if user_id not in currently_visible_ids and (
                    now_ts - stats["last_seen_ts"] >= VISIBILITY_LOST_TIMEOUT_SEC
                ):
                    session_sec = now_ts - stats["visible_since"]
                    stats["total_sec"] += max(0.0, session_sec)
                    stats["visible_since"] = None

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
                cv2.putText(
                    display_frame,
                    face_item["name"],
                    (left, max(20, top - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2,
                )
                cv2.putText(
                    display_frame,
                    f"time {visible_text}",
                    (left, bottom + 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

            stale_ids = [
                user_id
                for user_id, info in recent_tracks.items()
                if frame_count - info["last_seen_frame"] > TRACK_MAX_MISSING_FRAMES
            ]
            for user_id in stale_ids:
                del recent_tracks[user_id]

            cv2.imshow("Camera", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            if key == ord("w"):
                print("\n=== EDIT NAME ===")
                for uid in known_faces:
                    print(f"{uid}: {known_faces[uid]['name']}")
                selected = input("Enter ID to rename: ").strip()
                if selected in known_faces:
                    new_name = input("Enter new name: ").strip()
                    known_faces[selected]["name"] = new_name
                    save_base()
                    known_encodings, known_ids, known_names = build_encodings_dict(known_faces)
                    print(f"[INFO] Name for {selected} updated to {new_name}")
                else:
                    print("[WARN] Invalid ID")

            if key == ord("d"):
                print("\n=== DELETE FACE ===")
                for uid in known_faces:
                    print(f"{uid}: {known_faces[uid]['name']}")
                selected = input("Enter ID to delete: ").strip()
                if selected in known_faces:
                    del known_faces[selected]
                    if selected in visibility_stats:
                        del visibility_stats[selected]
                    if selected in recent_tracks:
                        del recent_tracks[selected]

                    face_dir = os.path.join(FACES_DIR, selected)
                    if os.path.isdir(face_dir):
                        shutil.rmtree(face_dir)
                        print(f"[INFO] Photo folder deleted: {face_dir}")

                    per_id_log = os.path.join(LOGS_DIR, f"{selected}.txt")
                    if os.path.exists(per_id_log):
                        os.remove(per_id_log)
                        print(f"[INFO] ID log deleted: {per_id_log}")

                    save_base()
                    known_encodings, known_ids, known_names = build_encodings_dict(known_faces)
                    print(f"[INFO] Face with ID {selected} deleted.")
                else:
                    print("[WARN] Invalid ID")

    finally:
        cap.release()
        cv2.destroyAllWindows()

        # Difference from low_main.py:
        # the worker must be signaled and joined so the extra process exits cleanly.
        try:
            task_queue.put_nowait(None)
        except queue.Full:
            try:
                task_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                task_queue.put_nowait(None)
            except queue.Full:
                pass

        worker.join(timeout=2.0)
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=1.0)

        final_ts = time.time()
        for user_id, stats in visibility_stats.items():
            if stats["visible_since"] is not None:
                stats["total_sec"] += max(0.0, final_ts - stats["visible_since"])
                stats["visible_since"] = None
            log_visibility_event(stats["name"], user_id, "Visibility zone summary", stats["total_sec"])
            print(f"[INFO] ID {user_id} was in visibility zone {format_duration(stats['total_sec'])}")
        print("[INFO] Completion. Database saved.")


if __name__ == "__main__":
    # Difference from low_main.py:
    # multiprocessing requires an explicit entry-point guard and start method.
    mp.set_start_method("spawn", force=True)
    main()
