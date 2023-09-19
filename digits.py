import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pyautogui

def extract_and_save_characters(image_path, output_folder):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, thresholded = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])  # Сортировка по координате X
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for idx, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        if w > 10 and h > 10:  # Отфильтровываем слишком маленькие контуры
            
            # Добавляем отступ
            x -= 20
            y -= 20
            w += 40
            h += 40
            
            # Убедимся, что координаты не выходят за пределы изображения
            x = max(x, 0)
            y = max(y, 0)
            w = min(w, img.shape[1] - x)
            h = min(h, img.shape[0] - y)
            
            character = img[y:y+h, x:x+w]
            filename = os.path.join(output_folder, f"{idx}.png")
            cv2.imwrite(filename, character)

# Загрузка данных MNIST
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Нормализация данных
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Добавление дополнительного измерения для канала цвета (только для grayscale изображений)
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# Определение архитектуры модели
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='sigmoid', input_shape=(28, 28, 1)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='sigmoid'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation='sigmoid'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# Компиляция модели
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Обучение модели
#model.fit(x_train, y_train, epochs=15, batch_size=64, validation_split=0.2)

# Оценка точности на тестовых данных
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

# Сохранение модели
#model.save('mnist_model.h5')

# Получить текущие значения ширины и высоты экрана и сохранить их в переменные screen_width и screen_height
screen_width, screen_height = pyautogui.size()

# Загрузка модели
model = tf.keras.models.load_model('mnist_model.h5')

digit_number = 1

# Цикл, который из папки "aaa" разделяет изображение на несколько в папку "bbb"
while os.path.isfile(f'D:\\aaa\\{digit_number}.png'):

    img_path = f'D:\\aaa\\{digit_number}.png'
    output_folder = f'D:\\bbb'
    
    # Вызвать функцию extract_and_save_characters() для обработки изображения img_path и сохранения результатов в output_folder
    extract_and_save_characters(img_path, output_folder)
    
    digit_number += 1
    
digit_number = 0

while os.path.isfile(f'D:\\bbb\\{digit_number}.png'):
    
    # Загрузка изображения
    img = cv2.imread(f'D:\\bbb\\{digit_number}.png', cv2.IMREAD_GRAYSCALE)
    
    # Подготовка изображения для модели
    img = 255 - img
    img = cv2.resize(img, (28, 28))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    
    # Предсказание с использованием модели
    prediction = model.predict(img)
    predicted_digit = np.argmax(prediction)
    probabilities = prediction[0]
    
    # Вывод результатов
    print(f"Предсказанная цифра: {predicted_digit}")
    for digit, probability in enumerate(probabilities):
        print(f"Вероятность для цифры {digit}: {probability:.4f}")
    
    # Отображение изображения
    plt.figure(figsize=(4, 4))
    plt.imshow(img[0, :, :, 0], cmap=plt.cm.binary)
    manager = plt.get_current_fig_manager()
    manager.window.wm_geometry("+{}+{}".format(screen_width - 400, screen_height - 500))
    plt.show()

    digit_number += 1
