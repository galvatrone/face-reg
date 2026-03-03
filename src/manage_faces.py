import argparse
import os
import pickle
import shutil

import numpy as np


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_FILE = os.path.join(PROJECT_DIR, "known_faces.pkl")
FACES_DIR = os.path.join(PROJECT_DIR, "faces")
LOGS_DIR = os.path.join(PROJECT_DIR, "logs")
MERGE_DISTANCE_THRESHOLD = 0.03


def load_database():
    if not os.path.exists(BASE_FILE):
        raise FileNotFoundError(f"Database not found: {BASE_FILE}")

    with open(BASE_FILE, "rb") as base_file:
        return pickle.load(base_file)


def save_database(face_db):
    with open(BASE_FILE, "wb") as base_file:
        pickle.dump(face_db, base_file)


def parse_args():
    parser = argparse.ArgumentParser(description="Rename, delete, or merge face IDs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    rename_parser = subparsers.add_parser("rename")
    rename_parser.add_argument("face_id")
    rename_parser.add_argument("new_name")

    delete_parser = subparsers.add_parser("delete")
    delete_parser.add_argument("face_id")

    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("source_id")
    merge_parser.add_argument("target_id")

    return parser.parse_args()


def ensure_face_id(face_db, face_id):
    if face_id not in face_db:
        raise KeyError(f"ID not found: {face_id}")


def add_unique_encodings(target_encodings, source_encodings):
    added_count = 0
    for encoding in source_encodings:
        encoding_array = np.asarray(encoding)
        if not target_encodings:
            target_encodings.append(encoding_array)
            added_count += 1
            continue

        distances = [
            float(np.linalg.norm(np.asarray(existing) - encoding_array))
            for existing in target_encodings
        ]
        if min(distances) > MERGE_DISTANCE_THRESHOLD:
            target_encodings.append(encoding_array)
            added_count += 1
    return added_count


def move_face_files(source_id, target_id):
    source_dir = os.path.join(FACES_DIR, source_id)
    target_dir = os.path.join(FACES_DIR, target_id)
    if not os.path.isdir(source_dir):
        return 0

    os.makedirs(target_dir, exist_ok=True)
    moved_count = 0
    for file_name in sorted(os.listdir(source_dir)):
        source_path = os.path.join(source_dir, file_name)
        if not os.path.isfile(source_path):
            continue

        target_path = os.path.join(target_dir, file_name)
        if os.path.exists(target_path):
            base_name, ext = os.path.splitext(file_name)
            target_path = os.path.join(target_dir, f"{base_name}_{source_id[:8]}{ext}")

        shutil.move(source_path, target_path)
        moved_count += 1

    shutil.rmtree(source_dir, ignore_errors=True)
    return moved_count


def remove_log(face_id):
    log_path = os.path.join(LOGS_DIR, f"{face_id}.txt")
    if os.path.exists(log_path):
        os.remove(log_path)


def rename_face(face_db, face_id, new_name):
    ensure_face_id(face_db, face_id)
    old_name = face_db[face_id].get("name", "")
    face_db[face_id]["name"] = new_name
    save_database(face_db)
    print(f"Renamed {face_id}: '{old_name}' -> '{new_name}'")


def delete_face(face_db, face_id):
    ensure_face_id(face_db, face_id)
    name = face_db[face_id].get("name", "")
    del face_db[face_id]
    save_database(face_db)

    face_dir = os.path.join(FACES_DIR, face_id)
    if os.path.isdir(face_dir):
        shutil.rmtree(face_dir, ignore_errors=True)
    remove_log(face_id)
    print(f"Deleted {face_id} ({name})")


def merge_faces(face_db, source_id, target_id):
    ensure_face_id(face_db, source_id)
    ensure_face_id(face_db, target_id)
    if source_id == target_id:
        raise ValueError("Source ID and target ID must be different.")

    source_data = face_db[source_id]
    target_data = face_db[target_id]
    source_encodings = list(source_data.get("encodings", []))
    target_encodings = target_data.setdefault("encodings", [])

    added_count = add_unique_encodings(target_encodings, source_encodings)
    moved_count = move_face_files(source_id, target_id)

    del face_db[source_id]
    save_database(face_db)
    remove_log(source_id)

    print(
        f"Merged {source_id} ({source_data.get('name', '')}) into "
        f"{target_id} ({target_data.get('name', '')}). "
        f"Added encodings: {added_count}, moved files: {moved_count}"
    )


def main():
    args = parse_args()
    face_db = load_database()

    if args.command == "rename":
        rename_face(face_db, args.face_id.strip(), args.new_name.strip())
    elif args.command == "delete":
        delete_face(face_db, args.face_id.strip())
    elif args.command == "merge":
        merge_faces(face_db, args.source_id.strip(), args.target_id.strip())


if __name__ == "__main__":
    main()
