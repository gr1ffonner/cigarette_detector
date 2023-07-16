import argparse
import os
import cv2
import numpy as np
from tensorflow.lite.python.interpreter import Interpreter


def tflite_detect_image(modelpath, image, lblpath, min_conf):
    # Загрузка меток классов
    with open(lblpath, "r") as f:
        labels = [line.strip() for line in f.readlines()]

    # Загрузка TensorFlow Lite модели
    interpreter = Interpreter(model_path=modelpath)
    interpreter.allocate_tensors()

    # Получение информации о входных и выходных тензорах модели
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]["shape"][1]
    width = input_details[0]["shape"][2]

    # Загрузка и предобработка изображения
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imH, imW, _ = image.shape
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)

    # Нормализация пикселей изображения, если используется не квантованная модель
    if input_details[0]["dtype"] == np.float32:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    # Выполнение обнаружения объектов на изображении
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()

    # Получение результатов обнаружения
    boxes = interpreter.get_tensor(output_details[1]["index"])[0]
    classes = interpreter.get_tensor(output_details[3]["index"])[0]
    scores = interpreter.get_tensor(output_details[0]["index"])[0]

    # Цикл по всем обнаруженным объектам и отрисовка bounding box-ов
    for i in range(len(scores)):
        if scores[i] > min_conf and scores[i] <= 1.0:
            ymin = int(max(1, (boxes[i][0] * imH)))
            xmin = int(max(1, (boxes[i][1] * imW)))
            ymax = int(min(imH, (boxes[i][2] * imH)))
            xmax = int(min(imW, (boxes[i][3] * imW)))
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    # Вывод изображения с bounding box-ами
    cv2.imshow("Object Detection", image)
    cv2.waitKey(1)


if __name__ == "__main__":
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", help="Path to the input video")
    args = parser.parse_args()

    # Определение путей и параметров модели
    MODEL_PATH = "detect.tflite"
    LABELS_PATH = "label_map.pbtxt"
    MIN_CONF_THRESHOLD = 0.05

    # Определение пути к видео из аргумента командной строки
    video_path = args.source

    # Загрузка видео
    video = cv2.VideoCapture(video_path)

    while True:
        # Чтение кадра из видео
        ret, frame = video.read()
        if not ret:
            break

        # Обнаружение объектов на кадре
        tflite_detect_image(MODEL_PATH, frame, LABELS_PATH, MIN_CONF_THRESHOLD)

        # Выход при нажатии клавиши 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Освобождение ресурсов
    video.release()
    cv2.destroyAllWindows()
