import os

import face_recognition


FACE_MATCH_THRESHOLD = 0.6
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(TOOLS_DIR, "input_photos")
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_first_face_encoding(image_path):
    image = face_recognition.load_image_file(image_path)
    locations = face_recognition.face_locations(image)
    if not locations:
        raise ValueError(f"Face not found in: {image_path}")

    encodings = face_recognition.face_encodings(image, known_face_locations=locations)
    if not encodings:
        raise ValueError(f"Failed to build face encoding: {image_path}")

    return encodings[0], locations[0]


def distance_to_confidence(distance, threshold=FACE_MATCH_THRESHOLD):
    if distance < 0.0:
        distance = 0.0
    if distance > 1.0:
        distance = 1.0

    confidence_range = max(1e-9, 1.0 - threshold)
    linear_value = (1.0 - distance) / (confidence_range * 2.0)
    linear_value = max(0.0, min(1.0, linear_value))

    if distance > threshold:
        return linear_value * 100.0

    boosted = linear_value + ((1.0 - linear_value) * pow((linear_value - 0.5) * 2.0, 0.2))
    boosted = max(0.0, min(1.0, boosted))
    return boosted * 100.0


def compare_encodings(encoding_1, encoding_2):
    distance = float(face_recognition.face_distance([encoding_1], encoding_2)[0])
    similarity_percent = distance_to_confidence(distance)
    is_match = distance <= FACE_MATCH_THRESHOLD
    return distance, similarity_percent, is_match


def ensure_input_dir():
    os.makedirs(INPUT_DIR, exist_ok=True)
    return INPUT_DIR


def resolve_input_path(image_arg):
    ensure_input_dir()
    if os.path.isabs(image_arg):
        return image_arg
    if os.path.exists(image_arg):
        return image_arg
    return os.path.join(INPUT_DIR, image_arg)


def get_first_two_input_photos():
    ensure_input_dir()
    photo_paths = []

    for name in sorted(os.listdir(INPUT_DIR)):
        path = os.path.join(INPUT_DIR, name)
        ext = os.path.splitext(name)[1].lower()
        if os.path.isfile(path) and ext in VALID_EXTENSIONS:
            photo_paths.append(path)

    if len(photo_paths) < 2:
        raise ValueError(
            f"Need at least 2 image files in input folder: {INPUT_DIR}"
        )

    return photo_paths[0], photo_paths[1]


def get_first_two_valid_faces():
    ensure_input_dir()
    valid_faces = []
    skipped_files = []
    checked_files = []

    for name in sorted(os.listdir(INPUT_DIR)):
        path = os.path.join(INPUT_DIR, name)
        ext = os.path.splitext(name)[1].lower()
        if not (os.path.isfile(path) and ext in VALID_EXTENSIONS):
            continue

        checked_files.append(path)

        try:
            encoding, location = load_first_face_encoding(path)
        except ValueError:
            skipped_files.append(path)
            continue

        valid_faces.append((path, encoding, location))
        if len(valid_faces) == 2:
            return valid_faces[0], valid_faces[1], skipped_files

    skipped_names = ", ".join(os.path.basename(path) for path in skipped_files) or "none"
    raise ValueError(
        "Need at least 2 image files with detectable faces in input folder: "
        f"{INPUT_DIR}. Checked: {len(checked_files)}, valid: {len(valid_faces)}, "
        f"skipped without face: {skipped_names}"
    )


def get_first_valid_face():
    ensure_input_dir()
    skipped_files = []
    checked_files = []

    for name in sorted(os.listdir(INPUT_DIR)):
        path = os.path.join(INPUT_DIR, name)
        ext = os.path.splitext(name)[1].lower()
        if not (os.path.isfile(path) and ext in VALID_EXTENSIONS):
            continue

        checked_files.append(path)

        try:
            encoding, location = load_first_face_encoding(path)
        except ValueError:
            skipped_files.append(path)
            continue

        return (path, encoding, location), skipped_files

    skipped_names = ", ".join(os.path.basename(path) for path in skipped_files) or "none"
    raise ValueError(
        "Need at least 1 image file with a detectable face in input folder: "
        f"{INPUT_DIR}. Checked: {len(checked_files)}, skipped without face: {skipped_names}"
    )
