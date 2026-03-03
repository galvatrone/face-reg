import os
import pickle

import numpy as np


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_FILE = os.path.join(PROJECT_DIR, "known_faces.pkl")
SIMILARITY_THRESHOLD = 0.03


def load_database():
    if not os.path.exists(BASE_FILE):
        raise FileNotFoundError(f"Database not found: {BASE_FILE}")

    with open(BASE_FILE, "rb") as base_file:
        return pickle.load(base_file)


def save_database(face_db):
    with open(BASE_FILE, "wb") as base_file:
        pickle.dump(face_db, base_file)


def dedupe_encodings(encodings):
    kept = []
    removed = 0
    for encoding in encodings:
        encoding_array = np.asarray(encoding)
        if not kept:
            kept.append(encoding_array)
            continue

        distances = [
            float(np.linalg.norm(np.asarray(existing) - encoding_array))
            for existing in kept
        ]
        if min(distances) <= SIMILARITY_THRESHOLD:
            removed += 1
            continue
        kept.append(encoding_array)

    return kept, removed


def main():
    face_db = load_database()
    total_removed = 0

    for face_id, data in sorted(face_db.items()):
        encodings = list(data.get("encodings", []))
        unique_encodings, removed = dedupe_encodings(encodings)
        if removed > 0:
            data["encodings"] = unique_encodings
            total_removed += removed
            print(
                f"{face_id} ({data.get('name', '')}): removed {removed}, "
                f"left {len(unique_encodings)}"
            )

    if total_removed == 0:
        print("No very similar duplicate encodings found.")
        return

    save_database(face_db)
    print(f"Optimization complete. Removed duplicate encodings: {total_removed}")


if __name__ == "__main__":
    main()
