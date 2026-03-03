import argparse
from collections import deque
import multiprocessing as mp
import os
import queue
import time

import cv2
import face_recognition
import numpy as np

from common import (
    FACE_MATCH_THRESHOLD,
    compare_encodings,
    distance_to_confidence,
    ensure_input_dir,
    get_first_valid_face,
    load_first_face_encoding,
    resolve_input_path,
)


WINDOW_NAME = "Camera vs Photo"
SMOOTHING_WINDOW_SEC = 5.0
DETECTION_SCALE = 0.5
DETECTION_FRAME_INTERVAL = 2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare a live camera face against one reference photo and show both in one window."
    )
    parser.add_argument(
        "image",
        nargs="?",
        help="Reference photo path. If omitted, the first valid image from input_photos is used.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Camera index for OpenCV (default: 0).",
    )
    return parser.parse_args()


def resize_to_height(image, target_height):
    if image.shape[0] == target_height:
        return image

    scale = target_height / image.shape[0]
    target_width = max(1, int(image.shape[1] * scale))
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)


def pick_largest_face(locations):
    return max(locations, key=lambda loc: (loc[2] - loc[0]) * (loc[1] - loc[3]))


def detection_worker(task_queue, result_queue):
    while True:
        task = task_queue.get()
        if task is None:
            break

        frame_id = task["frame_id"]
        small_frame = task["small_frame"]
        rgb_small_frame = small_frame[:, :, ::-1].copy()
        locations = face_recognition.face_locations(rgb_small_frame)

        if not locations:
            result_queue.put({"frame_id": frame_id, "face_location_small": None, "face_encoding": None})
            continue

        face_location_small = pick_largest_face(locations)
        encoding = face_recognition.face_encodings(
            rgb_small_frame,
            [face_location_small],
            num_jitters=1,
            model="small",
        )
        face_encoding = encoding[0] if encoding else None
        result_queue.put(
            {
                "frame_id": frame_id,
                "face_location_small": face_location_small,
                "face_encoding": face_encoding,
            }
        )


def drain_latest_result(result_queue):
    latest = None
    while True:
        try:
            latest = result_queue.get_nowait()
        except queue.Empty:
            return latest


def draw_reference_panel(reference_image, similarity_percent, is_match, has_live_face):
    panel = reference_image.copy()
    status_text = "MATCH" if is_match else "NO MATCH"
    color = (0, 180, 0) if is_match else (0, 0, 220)
    if not has_live_face:
        status_text = "NO FACE IN CAMERA"
        color = (0, 140, 255)

    overlay = panel.copy()
    cv2.rectangle(overlay, (0, 0), (panel.shape[1], 92), (255, 255, 255), -1)
    panel = cv2.addWeighted(overlay, 0.75, panel, 0.25, 0)

    cv2.putText(
        panel,
        f"Similarity: {similarity_percent:.2f}%",
        (20, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        status_text,
        (20, 74),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        "Reference Photo",
        (20, panel.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return panel


def draw_camera_panel(frame, face_location, similarity_percent, is_match):
    panel = frame.copy()
    if face_location is None:
        cv2.putText(
            panel,
            "Show one face to the camera",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 140, 255),
            2,
            cv2.LINE_AA,
        )
    else:
        top, right, bottom, left = face_location
        color = (0, 180, 0) if is_match else (0, 0, 220)
        cv2.rectangle(panel, (left, top), (right, bottom), color, 2)
        cv2.putText(
            panel,
            f"{similarity_percent:.2f}%",
            (left, max(30, top - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        panel,
        "Live Camera",
        (20, panel.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return panel


def build_combined_view(camera_panel, reference_panel):
    target_height = max(camera_panel.shape[0], reference_panel.shape[0])
    camera_panel = resize_to_height(camera_panel, target_height)
    reference_panel = resize_to_height(reference_panel, target_height)

    spacer = 16
    canvas_width = camera_panel.shape[1] + reference_panel.shape[1] + spacer
    canvas = np.full((target_height, canvas_width, 3), 24, dtype=np.uint8)

    canvas[:, 0 : camera_panel.shape[1]] = camera_panel
    start_x = camera_panel.shape[1] + spacer
    canvas[:, start_x : start_x + reference_panel.shape[1]] = reference_panel
    return canvas


def load_reference_photo(image_arg):
    skipped_files = []
    if image_arg:
        image_path = resolve_input_path(image_arg)
        encoding, location = load_first_face_encoding(image_path)
        return image_path, encoding, location, skipped_files

    face_item, skipped_files = get_first_valid_face()
    image_path, encoding, location = face_item
    return image_path, encoding, location, skipped_files


def scale_face_location(face_location_small, frame_shape):
    if face_location_small is None:
        return None

    scale_inv = 1.0 / DETECTION_SCALE
    top, right, bottom, left = face_location_small
    top = max(0, int(top * scale_inv))
    right = min(frame_shape[1], int(right * scale_inv))
    bottom = min(frame_shape[0], int(bottom * scale_inv))
    left = max(0, int(left * scale_inv))
    return (top, right, bottom, left)


def main():
    args = parse_args()
    input_dir = ensure_input_dir()
    image_path, reference_encoding, reference_location, skipped_files = load_reference_photo(args.image)

    reference_image = cv2.imread(image_path)
    if reference_image is None:
        raise ValueError(f"Failed to open reference image: {image_path}")

    top, right, bottom, left = reference_location
    cv2.rectangle(reference_image, (left, top), (right, bottom), (0, 180, 0), 2)

    capture = cv2.VideoCapture(args.camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open camera with index {args.camera_index}")

    task_queue = mp.Queue(maxsize=1)
    result_queue = mp.Queue(maxsize=1)
    worker = mp.Process(target=detection_worker, args=(task_queue, result_queue), daemon=True)
    worker.start()

    similarity_percent = 0.0
    is_match = False
    face_location = None
    recent_distances = deque()
    frame_count = 0
    pending_frame_id = None

    print(f"Input folder: {input_dir}")
    print(f"Reference photo: {os.path.abspath(image_path)}")
    if skipped_files:
        print("Skipped files without detectable face before selecting reference:")
        for skipped_path in skipped_files:
            print(f"  {os.path.abspath(skipped_path)}")
    print("Press 'q' or Esc to exit.")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("Failed to read frame from camera.")
                break

            frame = cv2.flip(frame, 1)

            latest_result = drain_latest_result(result_queue)
            if latest_result is not None:
                pending_frame_id = None
                face_location = scale_face_location(latest_result["face_location_small"], frame.shape)
                if latest_result["face_encoding"] is not None:
                    distance, _, _ = compare_encodings(
                        reference_encoding,
                        latest_result["face_encoding"],
                    )
                    now = time.monotonic()
                    recent_distances.append((now, distance))
                    while recent_distances and now - recent_distances[0][0] > SMOOTHING_WINDOW_SEC:
                        recent_distances.popleft()

                    avg_distance = sum(item[1] for item in recent_distances) / len(recent_distances)
                    similarity_percent = distance_to_confidence(avg_distance)
                    is_match = avg_distance <= FACE_MATCH_THRESHOLD
                else:
                    similarity_percent = 0.0
                    is_match = False
                    recent_distances.clear()
            elif face_location is None:
                similarity_percent = 0.0
                is_match = False

            if frame_count % DETECTION_FRAME_INTERVAL == 0 and pending_frame_id is None:
                small_frame = cv2.resize(frame, (0, 0), fx=DETECTION_SCALE, fy=DETECTION_SCALE)
                payload = {"frame_id": frame_count, "small_frame": small_frame}
                try:
                    task_queue.put_nowait(payload)
                    pending_frame_id = frame_count
                except queue.Full:
                    pass

            frame_count += 1

            camera_panel = draw_camera_panel(frame, face_location, similarity_percent, is_match)
            reference_panel = draw_reference_panel(
                reference_image,
                similarity_percent,
                is_match,
                face_location is not None,
            )
            combined_view = build_combined_view(camera_panel, reference_panel)

            cv2.imshow(WINDOW_NAME, combined_view)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
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


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
