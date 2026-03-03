import os
import pickle


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_FILE = os.path.join(PROJECT_DIR, "known_faces.pkl")


def main():
    if not os.path.exists(BASE_FILE):
        return

    with open(BASE_FILE, "rb") as base_file:
        face_db = pickle.load(base_file)

    for face_id, data in sorted(face_db.items(), key=lambda item: str(item[1].get("name", "")).casefold()):
        name = str(data.get("name", "")).replace("\t", " ").strip()
        encodings = data.get("encodings", [])
        print(f"{face_id}\t{name}\t{len(encodings)}")


if __name__ == "__main__":
    main()
