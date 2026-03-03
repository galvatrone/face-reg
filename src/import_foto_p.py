import os
import pickle
import shutil
import uuid

import cv2
import face_recognition


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_FILE = os.path.join(PROJECT_DIR, "known_faces.pkl")
FACES_DIR = os.path.join(PROJECT_DIR, "faces")
IMPORT_DIR = os.path.join(PROJECT_DIR, "foto_p")
RESULT_DIR = os.path.join(PROJECT_DIR, "foto_p_result")

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_database():
    if os.path.exists(BASE_FILE):
        with open(BASE_FILE, "rb") as f:
            return pickle.load(f)
    return {}


def save_database(face_db):
    with open(BASE_FILE, "wb") as f:
        pickle.dump(face_db, f)


def ensure_dirs():
    os.makedirs(FACES_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    if not os.path.exists(IMPORT_DIR):
        os.makedirs(IMPORT_DIR, exist_ok=True)
        print(f"[INFO] Created import folder: {IMPORT_DIR}")
        print("[INFO] Add subfolders like foto_p/Misha/photo1.jpg and run again.")
        return False
    return True


def list_person_folders():
    person_dirs = []
    for name in sorted(os.listdir(IMPORT_DIR)):
        path = os.path.join(IMPORT_DIR, name)
        if os.path.isdir(path):
            person_dirs.append((name, path))
    return person_dirs


def list_photo_files(folder_path):
    files = []
    for name in sorted(os.listdir(folder_path)):
        path = os.path.join(folder_path, name)
        ext = os.path.splitext(name)[1].lower()
        if os.path.isfile(path) and ext in VALID_EXTENSIONS:
            files.append(path)
    return files


def get_face_dir(user_id):
    face_dir = os.path.join(FACES_DIR, user_id)
    os.makedirs(face_dir, exist_ok=True)
    return face_dir


def has_primary_photo(face_dir):
    for name in os.listdir(face_dir):
        if os.path.isfile(os.path.join(face_dir, name)) and name.startswith("main."):
            return True
    return False


def save_source_copy(face_dir, source_path, is_first):
    ext = os.path.splitext(source_path)[1].lower() or ".jpg"
    filename = f"main{ext}" if is_first else f"{uuid.uuid4().hex[:8]}{ext}"
    target_path = os.path.join(face_dir, filename)
    shutil.copy2(source_path, target_path)
    return target_path


def get_result_dir(person_name):
    result_dir = os.path.join(RESULT_DIR, person_name)
    os.makedirs(result_dir, exist_ok=True)
    return result_dir


def save_annotated_preview(person_name, source_path, face_location):
    image = cv2.imread(source_path)
    if image is None:
        print(f"[WARN] Failed to build preview for {source_path}: OpenCV could not read file")
        return None

    top, right, bottom, left = face_location
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 3)

    result_dir = get_result_dir(person_name)
    base_name, ext = os.path.splitext(os.path.basename(source_path))
    ext = ext or ".jpg"
    preview_path = os.path.join(result_dir, f"{base_name}_boxed{ext}")
    cv2.imwrite(preview_path, image)
    return preview_path


def extract_encoding(image_path):
    try:
        image = face_recognition.load_image_file(image_path)
    except Exception as exc:
        print(f"[WARN] Failed to read {image_path}: {exc}")
        return None

    locations = face_recognition.face_locations(image, model="hog")
    if len(locations) != 1:
        print(f"[WARN] Skipped {image_path}: expected 1 face, found {len(locations)}")
        return None

    encodings = face_recognition.face_encodings(image, locations, num_jitters=1, model="small")
    if not encodings:
        print(f"[WARN] Skipped {image_path}: could not build encoding")
        return None

    return encodings[0], locations[0]


def find_existing_user_id(face_db, person_name):
    wanted = person_name.strip().casefold()
    for user_id, data in face_db.items():
        if str(data.get("name", "")).strip().casefold() == wanted:
            return user_id
    return None


def import_person_folder(face_db, person_name, folder_path):
    photo_files = list_photo_files(folder_path)
    if not photo_files:
        print(f"[WARN] Skipped '{person_name}': no supported image files")
        return False

    existing_user_id = find_existing_user_id(face_db, person_name)
    user_id = existing_user_id or str(uuid.uuid4())
    face_dir = get_face_dir(user_id)
    if existing_user_id:
        encodings = face_db[user_id].setdefault("encodings", [])
        face_db[user_id]["name"] = person_name
        print(f"[INFO] Found existing person '{person_name}' as ID {user_id}. Appending encodings.")
    else:
        encodings = []
    copied_count = 0
    primary_exists = has_primary_photo(face_dir)

    for photo_path in photo_files:
        result = extract_encoding(photo_path)
        if result is None:
            continue
        encoding, face_location = result

        encodings.append(encoding)
        copied_path = save_source_copy(
            face_dir,
            photo_path,
            is_first=(copied_count == 0 and not primary_exists),
        )
        preview_path = save_annotated_preview(person_name, photo_path, face_location)
        copied_count += 1
        primary_exists = True
        print(f"[INFO] Added encoding from {photo_path} -> {copied_path}")
        if preview_path:
            print(f"[INFO] Preview with box saved -> {preview_path}")

    if copied_count == 0:
        if not existing_user_id:
            try:
                os.rmdir(face_dir)
            except OSError:
                pass
        print(f"[WARN] Skipped '{person_name}': no valid face encodings")
        return False

    if not existing_user_id:
        face_db[user_id] = {
            "name": person_name,
            "encodings": encodings,
        }
        print(f"[INFO] Imported '{person_name}' as ID {user_id} with {len(encodings)} encoding(s)")
    else:
        print(
            f"[INFO] Updated '{person_name}' (ID {user_id}): total {len(encodings)} encoding(s)"
        )
    return True


def main():
    if not ensure_dirs():
        return

    person_folders = list_person_folders()
    if not person_folders:
        print(f"[INFO] No person folders found in {IMPORT_DIR}")
        return

    face_db = load_database()
    imported_count = 0

    for person_name, folder_path in person_folders:
        if import_person_folder(face_db, person_name, folder_path):
            imported_count += 1

    if imported_count == 0:
        print("[INFO] Nothing imported.")
        return

    save_database(face_db)
    print(f"[INFO] Import complete. Added {imported_count} person(s).")


if __name__ == "__main__":
    main()
