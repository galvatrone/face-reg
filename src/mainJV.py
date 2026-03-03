import os
import queue
import time
import multiprocessing as mp

import cv2

import main as core


FRAME_DIR = os.path.join(core.PROJECT_DIR, "ui_frames")
FRAME_FILE = os.path.join(FRAME_DIR, "mainJV_frame.jpg")
STATUS_FILE = os.path.join(FRAME_DIR, "mainJV_status.txt")
FRAME_WRITE_INTERVAL = 0.05
JPEG_QUALITY = 75


def ensure_ui_dirs():
    os.makedirs(FRAME_DIR, exist_ok=True)


def write_status(text):
    ensure_ui_dirs()
    with open(STATUS_FILE, "w", encoding="utf-8") as status_file:
        status_file.write(text.strip() + "\n")


def write_frame(frame):
    ensure_ui_dirs()
    temp_file = FRAME_FILE + ".tmp"
    ok, encoded = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
    )
    if not ok:
        return

    with open(temp_file, "wb") as frame_file:
        frame_file.write(encoded.tobytes())
    os.replace(temp_file, FRAME_FILE)


def main():
    core.cv2.setNumThreads(os.cpu_count() or 1)
    ensure_ui_dirs()
    write_status("Starting")

    cap = core.open_camera()
    if cap is None or not cap.isOpened():
        message = "Failed to open camera"
        print(message)
        write_status(message)
        return

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, core.CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, core.CAPTURE_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, core.CAPTURE_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    task_queue = mp.Queue(maxsize=core.WORKER_COUNT)
    result_queue = mp.Queue(maxsize=core.WORKER_COUNT * 2)
    workers = [
        mp.Process(target=core.detection_worker, args=(task_queue, result_queue), daemon=True)
        for _ in range(core.WORKER_COUNT)
    ]
    for worker in workers:
        worker.start()

    frame_count = 0
    active_faces = []
    recent_tracks = {}
    last_encoding_add_frame = {}
    visibility_stats = {}
    frame_cache = {}
    pending_frame_ids = set()
    last_frame_write = 0.0

    print(f"[INFO] UI frame file: {FRAME_FILE}")
    print(f"[INFO] Status file: {STATUS_FILE}")
    print(f"[INFO] Detection workers: {core.WORKER_COUNT}")
    print("[INFO] Press Ctrl+C in terminal to stop.")
    write_status("Running")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame reading error")
                write_status("Frame reading error")
                break

            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()

            pending_results = core.drain_pending_results(result_queue)
            if pending_results:
                for result in pending_results:
                    processed_frame = frame_cache.pop(result["frame_id"], None)
                    pending_frame_ids.discard(result["frame_id"])
                    if processed_frame is None:
                        continue
                    active_faces = core.process_detection_result(
                        result,
                        processed_frame,
                        result["frame_id"],
                        active_faces,
                        recent_tracks,
                        last_encoding_add_frame,
                    )
            elif active_faces and all(face_item.get("held") for face_item in active_faces):
                active_faces = core.build_held_faces(frame_count, recent_tracks)

            if frame_count % core.DETECTION_FRAME_INTERVAL == 0 and len(pending_frame_ids) < core.WORKER_COUNT:
                small_frame = cv2.resize(frame, (0, 0), fx=core.DETECTION_SCALE, fy=core.DETECTION_SCALE)
                payload = {"frame_id": frame_count, "small_frame": small_frame}
                try:
                    task_queue.put_nowait(payload)
                    frame_cache[frame_count] = frame.copy()
                    pending_frame_ids.add(frame_count)
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
                    now_ts - stats["last_seen_ts"] >= core.VISIBILITY_LOST_TIMEOUT_SEC
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
                    visible_text = core.format_duration(elapsed_sec)

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
                if frame_count - info["last_seen_frame"] > core.TRACK_MAX_MISSING_FRAMES
            ]
            for user_id in stale_ids:
                del recent_tracks[user_id]

            cv2.putText(
                display_frame,
                "mainJV.py -> Java UI frame",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            if now_ts - last_frame_write >= FRAME_WRITE_INTERVAL:
                write_frame(display_frame)
                faces_text = f"Running | faces: {len(active_faces)} | frame: {frame_count}"
                write_status(faces_text)
                last_frame_write = now_ts

    except KeyboardInterrupt:
        print("[INFO] Stopped by user.")
        write_status("Stopped by user")
    finally:
        cap.release()

        for _ in workers:
            sent_stop = False
            while not sent_stop:
                try:
                    task_queue.put_nowait(None)
                    sent_stop = True
                except queue.Full:
                    try:
                        task_queue.get_nowait()
                    except queue.Empty:
                        sent_stop = True

        for worker in workers:
            worker.join(timeout=2.0)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=1.0)

        final_ts = time.time()
        for user_id, stats in visibility_stats.items():
            if stats["visible_since"] is not None:
                stats["total_sec"] += max(0.0, final_ts - stats["visible_since"])
                stats["visible_since"] = None
            core.log_visibility_event(
                stats["name"],
                user_id,
                "Visibility zone summary",
                stats["total_sec"],
            )
            print(f"[INFO] ID {user_id} was in visibility zone {core.format_duration(stats['total_sec'])}")

        if os.path.exists(FRAME_FILE):
            write_status("Stopped")
        print("[INFO] Completion. Database saved.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
