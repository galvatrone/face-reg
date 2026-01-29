import requests
import os
import random
import shutil

API_KEY = "49716050-9c27b213331b439fa3b7175be"
test_folder = r'C:\AllCodes\Neuran test\Python\test_images'

if not os.path.exists(test_folder):
    os.makedirs(test_folder)

def download_images(query, count=15):
    response = requests.get(f"https://pixabay.com/api/?key={API_KEY}&q={query}&image_type=photo&per_page={count}")
    image_paths = []

    if response.status_code == 200:
        images = response.json()['hits']
        for i, image in enumerate(images):
            file_name = f"{query}_{i+1}.jpg"
            path = os.path.join(test_folder, file_name)
            img_data = requests.get(image['webformatURL']).content
            with open(path, 'wb') as f:
                f.write(img_data)
            image_paths.append(path)
    else:
        print(f"Ошибка запроса для {query}: {response.status_code}")
    
    return image_paths

# Скачиваем изображения
dog_images = download_images("dogs", 15)
cat_images = download_images("cats", 15)

# Перемешиваем
all_images = dog_images + cat_images
random.shuffle(all_images)

# Переименовываем файлы в перемешанном порядке
for i, old_path in enumerate(all_images):
    new_path = os.path.join(test_folder, f"test_image_{i+1}.jpg")
    os.rename(old_path, new_path)

print(f"✅ Загружено и перемешано {len(all_images)} изображений в {test_folder}")