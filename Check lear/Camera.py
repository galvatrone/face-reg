import cv2
import face_recognition

# Открываем камеру с флагом CAP_DSHOW для Windows (устранение проблем с драйверами)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Устанавливаем разрешение камеры (можно изменить по желанию)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_count = 0  # Счётчик кадров

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Если не удалось получить кадр — выходим

    frame = cv2.flip(frame, 0)  # Зеркалим кадр (удобно для пользователя)
    frame = cv2.flip(frame, 1)  # Зеркалим кадр (удобно для пользователя)

    # Уменьшаем размер кадра для ускорения распознавания лиц
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)  # 1/4 от оригинального размера
    rgb_small_frame = small_frame[:, :, ::-1]  # Перевод из BGR в RGB

    # Распознаём лица только на каждом втором кадре (ускоряет работу)
    if frame_count % 2 == 0:
        # Используем модель HOG — быстрее, чем CNN
        face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")

        # Масштабируем координаты обратно к оригинальному размеру кадра
        scaled_locations = [(top*4, right*4, bottom*4, left*4)
                            for top, right, bottom, left in face_locations]

    frame_count += 1  # Увеличиваем счётчик кадров

    # Отображаем найденные лица на оригинальном кадре
    if 'scaled_locations' in locals():
        for top, right, bottom, left in scaled_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    # Показываем кадр в окне
    cv2.imshow('Camera', frame)

    # Выход при нажатии клавиши 'q'
    if cv2.waitKey(1) == ord('q'):
        break

# Освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()
