from PIL import Image
import numpy as np
# import matplotlib.pyplot as plt #для отрисовки изображений
import random
import sys #для работы с путями
sys.stdout.reconfigure(encoding='utf-8') #для работы с путями
import os   #для работы с путями
import time #для задержки

class NeuralNetwork:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        
        # Инициализация весов
        self.weights_input_hidden1 = np.random.uniform(-1, 1, (input_size, hidden_size1))
        self.weights_hidden1_hidden2 = np.random.uniform(-1, 1, (hidden_size1, hidden_size2))
        self.weights_hidden2_output = np.random.uniform(-1, 1, (hidden_size2, output_size))

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))
    
    def load_image_as_input_array(path, size=(28, 28)):
        # Открываем изображение и переводим в оттенки серого
        image = Image.open(path).convert('L')
        # Меняем размер
        image = image.resize(size)
        # Преобразуем в numpy массив и нормализуем
        image_array = np.array(image) / 255.0
        # Плоский массив (1D) — это будет вход для нейросети
        return image_array.flatten()
    
    @staticmethod
    def load_dataset_from_folder(folder_path, size=(28, 28)):
        images = []

        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                img_path = os.path.join(folder_path, filename)

                # Открываем и приводим к нужному размеру
                img = Image.open(img_path).convert('L')  # 'L' — grayscale
                img = img.resize(size)

                # Преобразуем в numpy и нормализуем от 0 до 1
                img_array = np.array(img) / 255.0
                img_array = img_array.flatten()  # В 1D-массив, если нужно

                images.append(img_array)

        return np.array(images)

    def forward(self, inputs):
        # Прямой проход
        hidden_inputs1 = np.dot(inputs, self.weights_input_hidden1)
        hidden_outputs1 = self.sigmoid(hidden_inputs1)

        hidden_inputs2 = np.dot(hidden_outputs1, self.weights_hidden1_hidden2)
        hidden_outputs2 = self.sigmoid(hidden_inputs2)

        final_inputs = np.dot(hidden_outputs2, self.weights_hidden2_output)
        final_outputs = self.sigmoid(final_inputs)

        return final_outputs

    # def load_weights_from_file(self, filename):
    #     with open(filename, 'r') as file:
    #         self.weights_input_hidden1 = np.array([list(map(float, file.readline().split())) for _ in range(self.input_size)])
    #         self.weights_hidden1_hidden2 = np.array([list(map(float, file.readline().split())) for _ in range(self.hidden_size1)])
    #         self.weights_hidden2_output = np.array([list(map(float, file.readline().split())) for _ in range(self.hidden_size2)])

    def load_weights_from_file(self, filename):
        try:
            with open(filename, 'r') as file:
                self.weights_input_hidden1 = np.array([list(map(float, file.readline().split())) for _ in range(self.input_size)])
                self.weights_hidden1_hidden2 = np.array([list(map(float, file.readline().split())) for _ in range(self.hidden_size1)])
                self.weights_hidden2_output = np.array([list(map(float, file.readline().split())) for _ in range(self.hidden_size2)])
        except FileNotFoundError:
            print(f"Файл {filename} не найден. Инициализируем веса случайными значениями.")
            self.weights_input_hidden1 = np.random.uniform(-1, 1, (self.input_size, self.hidden_size1))
            self.weights_hidden1_hidden2 = np.random.uniform(-1, 1, (self.hidden_size1, self.hidden_size2))
            self.weights_hidden2_output = np.random.uniform(-1, 1, (self.hidden_size2, self.output_size))

    

    def save_weights_to_file(self, filename):
        try:
            with open(filename, 'w') as file:
                # Сохранение весов в файл
                # print(f"Сохранение весов в файл {filename}...")
                np.savetxt(file, self.weights_input_hidden1)
                np.savetxt(file, self.weights_hidden1_hidden2)
                np.savetxt(file, self.weights_hidden2_output)
        except FileNotFoundError:
            print(f"Не удалось сохранить веса в файл {filename}. Проверьте права доступа.")
            create_file = input("Создать файл? (y/n): ")
            if create_file.lower() == 'y':
                with open(filename, 'w') as file:
                    np.savetxt(file, self.weights_input_hidden1)
                    np.savetxt(file, self.weights_hidden1_hidden2)
                    np.savetxt(file, self.weights_hidden2_output)
                print(f"Файл {filename} успешно создан.")
            else:
                print("Файл не создан. Проверьте права доступа.")

    def train(self, inputs, targets, learning_rate=0.1, epochs=1000000):
        for epoch in range(epochs):
            total_error = 0.0
            for input_example, target_example in zip(inputs, targets):
                # Прямой проход
                hidden_inputs1 = np.dot(input_example, self.weights_input_hidden1)
                hidden_outputs1 = self.sigmoid(hidden_inputs1)

                hidden_inputs2 = np.dot(hidden_outputs1, self.weights_hidden1_hidden2)
                hidden_outputs2 = self.sigmoid(hidden_inputs2)

                final_inputs = np.dot(hidden_outputs2, self.weights_hidden2_output)
                final_outputs = self.sigmoid(final_inputs)

                # Обратное распространение ошибки
                output_errors = target_example - final_outputs
                output_deltas = output_errors * final_outputs * (1 - final_outputs)

                hidden_errors2 = np.dot(output_deltas, self.weights_hidden2_output.T)
                hidden_deltas2 = hidden_errors2 * hidden_outputs2 * (1 - hidden_outputs2)

                hidden_errors1 = np.dot(hidden_deltas2, self.weights_hidden1_hidden2.T)
                hidden_deltas1 = hidden_errors1 * hidden_outputs1 * (1 - hidden_outputs1)

                # Обновление весов
                self.weights_hidden1_hidden2 += learning_rate * np.outer(hidden_outputs1, hidden_deltas2)
                self.weights_input_hidden1 += learning_rate * np.outer(input_example, hidden_deltas1)
                self.weights_hidden2_output += learning_rate * np.outer(hidden_outputs2, output_deltas)

                # Суммарная ошибка
                total_error += np.sum(output_errors ** 2)

            if epoch % 500 == 0:
                print(f"Эпоха: {epoch}, Ошибка: {total_error}, Скорость обучения: {learning_rate}")
                self.save_weights_to_file("PYweights.txt")

            if epoch % 10000 == 0:
                self.save_weights_to_file("PYweights.txt")
                learning_rate *= 1.00009

            if total_error < 0.01:
                print(f"Обучение завершено. Эпоха: {epoch}, Ошибка: {total_error}")
                self.save_weights_to_file("PYweights.txt")
                break


if __name__ == "__main__":
    input_size = 784
    hidden_size1 = 128
    hidden_size2 = 64
    output_size = 2

    neural_net = NeuralNetwork(input_size, hidden_size1, hidden_size2, output_size)
    neural_net.load_weights_from_file("PYweights.txt")



    folder_path =r'C:\AllCodes\Neuran test\Python\images' # Папка с изображениями

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Папка {folder_path} создана.")

    # Пример входных данных и целевых значений
    inputs = neural_net.load_dataset_from_folder(folder_path, size=(28, 28))
    # inputs = [
    #     [0, 0, 1, 1], [1, 1, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1],
    #     [0, 1, 1, 0], [0, 1, 1, 1], [0, 1, 0, 0], [0, 1, 2, 3], [0, 2, 2, 1],
    #     [1, 1, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1], [1, 0, 2, 0], [0, 0, 1, 0],
    #     [2, 0, 3, 3], [1, 0, 0, 4], [1, 1, 2, 2], [0, 2, 2, 2], [1, 1, 3, 3],
    #     [1, 2, 3, 3], [2, 2, 2, 2], [4, 4, 4, 4], [5, 5, 5, 5], [6, 4, 6, 6]
    # ]
    
    # Допустим, 3 класса: "Dogs", "Cats"
    class_names = ["Dogs", "Cats"]

    # Тогда метки в one-hot виде:30 cats and 30 dogs
    targets = [
        [0, 1],  # Cats
        [0, 1],  # Cats
        [0, 1],  # Cats
        [0, 1],  # Cats
        [0, 1],  # Cats
        [0, 1],  # Cats
        [0, 1],  # Cats
        [0, 1],  # Cats
        [0, 1],  # Cats
        [0, 1],  # Cats
        [0, 1],  # Cats
        [0, 1],  # Cats
        [0, 1],  # Cats
        [0, 1],  # Cats
        [0, 1],  # Cats
        [0, 1],  # Cats
        [0, 1],  # Cats
        [0, 1],  # Cats
        [0, 1],  # Cats
        [0, 1],  # Cats
        [0, 1],  # Cats
        [0, 1],  # Cats
        [0, 1],  # Cats
        [0, 1],  # Cats
        [0, 1],  # Cats
        [0, 1],  # Cats
        [0, 1],  # Cats
        [0, 1],  # Cats
        [0, 1],  # Cats
        [0, 1],  # Cats
        [0, 1],  # Cats
        [1, 0],  # Dogs
        [1, 0],  # Dogs
        [1, 0],  # Dogs
        [1, 0],  # Dogs
        [1, 0],  # Dogs[
        [1, 0],  # Dogs
        [1, 0],  # Dogs
        [1, 0],  # Dogs
        [1, 0],  # Dogs
        [1, 0],  # Dogs
        [1, 0],  # Dogs
        [1, 0],  # Dogs
        [   1, 0],  # Dogs
        [1, 0],  # Dogs
        [1, 0],  # Dogs
        [1, 0],  # Dogs
        [1, 0],  # Dogs
        [1, 0],  # Dogs
        [1, 0],  # Dogs
        [1, 0],  # Dogs
        [1, 0],  # Dogs
        [1, 0],  # Dogs
        [1, 0],  # Dogs
        [1, 0],  # Dogs
        [1, 0],  # Dogs
        [1, 0],  # Dogs]
        [1, 0],  # Dogs
        [1, 0],  # Dogs
        [1, 0],  # Dogs
        [1, 0],  # Dogs
        ...
    ]


    # Обучение сети
    # neural_net.train(inputs, targets)


