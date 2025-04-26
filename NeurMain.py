import cv2
import face_recognition
import pickle
import numpy as np
import os
import dlib
import uuid

# Абсолютный путь к папке проекта
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Папка проекта и моделей

MODELS_DIR = os.path.join(PROJECT_DIR, "models")


predictor_path = os.path.join(MODELS_DIR, "shape_predictor_68_face_landmarks.dat")
rec_model_path = os.path.join(MODELS_DIR, "dlib_face_recognition_resnet_model_v1.dat")

predictor = dlib.shape_predictor(predictor_path)
face_rec_model = dlib.face_recognition_model_v1(rec_model_path)

# Папка с базой и фотографиями
BASE_FILE = os.path.join(PROJECT_DIR, "known_faces.pkl")
FACES_DIR = os.path.join(PROJECT_DIR, "faces")


def log_face_added(name, face_id, encoding, total_faces, log_file=None):
    if log_file is None:
        log_file = os.path.join(PROJECT_DIR, "log.txt")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"Добавлено лицо: {name} (ID: {face_id})\n")
        f.write("Кодировка:\n")
        f.write(f"{encoding}\n")
        f.write(f"Всего ID в базе: {total_faces}\n")
        f.write("-" * 80 + "\n")




# Создание папки для фотографий
if not os.path.exists(FACES_DIR):
    os.makedirs(FACES_DIR)

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

# Камера
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Не удалось открыть камеру")
    exit()

frame_count = 0
scaled_locations, scaled_encodings = [], []

print("[INFO] Нажми 'q' для выхода, 'w' — редактировать имя, 'd' — удалить лицо.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Ошибка чтения кадра")
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.flip(frame, 0)
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

            rect = dlib.rectangle(left, top, right, bottom)
            shape = predictor(frame, rect)
            descriptor = face_rec_model.compute_face_descriptor(frame, shape)
            face_encoding = np.array(descriptor)
            scaled_encodings.append(face_encoding)

    frame_count += 1

    for face_encoding, (top, right, bottom, left) in zip(scaled_encodings, scaled_locations):
        name = "Unknown"

        if known_encodings:
            distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_index = np.argmin(distances)

            if distances[best_index] < 0.6:
                matched_id = known_ids[best_index]  # ЭТО ВАЖНО

                if matched_id in known_faces:
                    name = known_faces[matched_id]["name"]
                else:
                    name = "Unknown"    # Если ID был удалён, выводим "Неизвестно"


                # Проверяем, существует ли matched_id в known_faces
                if matched_id in known_faces:
                    name = known_faces[matched_id]["name"]
                else:
                    name = "Unknown"  # Если ID был удалён, выводим "Неизвестно"
            else:
                print("\n===ДОБАВЛЕНИЕ ИМЕНИ ===")

                # Добавляем новое лицо с UUID
                new_id = str(uuid.uuid4())  # Генерация уникального ID
                known_faces[new_id] = {
                    "name": "Unknown",
                    "encodings": [face_encoding]
                }
                face_img = frame[top:bottom, left:right]
                face_path = os.path.join(FACES_DIR, f"{new_id}.jpg")
                cv2.imwrite(face_path, face_img)

                with open(BASE_FILE, "wb") as f:
                    pickle.dump(known_faces, f)
                # Логируем добавление лица
                log_face_added(name, new_id, face_encoding, len(known_faces))

                known_encodings, known_ids, known_names = build_encodings_dict(known_faces)
                print(f"[INFO] Добавлено новое лицо: {new_id}, фото сохранено как {face_path}")

                name = "Unknown"

        # Рисуем лицо
        cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(display_frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

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
            with open(BASE_FILE, "wb") as f:
                pickle.dump(known_faces, f)
            
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

            # Удаляем фотографию лица
            face_path = os.path.join(FACES_DIR, f"{selected}.jpg")
            if os.path.exists(face_path):
                os.remove(face_path)
                print(f"[INFO] Файл {selected}.jpg удалён.")
            with open(BASE_FILE, "wb") as f:
                pickle.dump(known_faces, f)
            print(f"[INFO] Лицо с ID {selected} удалено.")

            # Перестроим вспомогательные списки
            known_encodings, known_ids, known_names = build_encodings_dict(known_faces)
        else:
            print("[WARN] Неверный ID")

cap.release()
cv2.destroyAllWindows()
print("[INFO] Завершение. База сохранена.")
