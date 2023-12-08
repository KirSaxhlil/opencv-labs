import cv2

face_cascade = cv2.CascadeClassifier('cascad_trained/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while False:
    # Чтение кадра из видеопотока
    ret, frame = cap.read()

    # Преобразование кадра в оттенки серого
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Обнаружение лиц в кадре
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Отрисовка прямоугольников вокруг обнаруженных лиц
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.putText(frame, 'Number of faces: ' + str(len(faces)), (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0),2)

    # Отображение результата
    cv2.imshow('Face Detection', frame)

    # Прерывание цикла при нажатии клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()

import numpy as np
import time
import sys
import os

CONFIDENCE = 0
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
# конфигурация нейронной сети
#config_path = "cfg/yolov3.cfg"
config_path = "cfg/yolov3-face.cfg"
# файл весов сети YOLO
#weights_path = "weights/yolov3.weights"
#weights_path = "weights/yolov3-tiny.weights"
weights_path = "weights/yolov3-wider_16000.weights" #FACES
# загрузка всех меток классов (объектов)
#labels = open("data/coco.names").read().strip().split("\n")
labels = open("data/face.names").read().strip().split("\n")
print(labels)
# генерируем цвета для каждого объекта и последующего построения
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

# загружаем сеть YOLO
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

path_name = "images/street.jpg"
while True:
    #image = cv2.imread(path_name)
    ret, image = cap.read()
    file_name = os.path.basename(path_name)
    filename, ext = file_name.split(".")

    h, w = image.shape[:2]
    # создать 4D blob
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)

    #print("image.shape:", image.shape)
    #print("blob.shape:", blob.shape)

    # устанавливает blob как вход сети
    net.setInput(blob)
    # получаем имена всех слоев
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
    #print(ln)
    # прямая связь (вывод) и получение выхода сети
    # измерение времени для обработки в секундах
    #start = time.perf_counter()
    layer_outputs = net.forward(ln)
    #time_took = time.perf_counter() - start
    #print(f"Потребовалось: {time_took:.2f}s")


    font_scale = 1
    thickness = 1
    boxes, confidences, class_ids = [], [], []
    # перебираем каждый из выходов слоя
    for output in layer_outputs:
        # перебираем каждое обнаружение объекта
        #print(len(output))
        for detection in output:
            # извлекаем идентификатор класса (метку) и достоверность (как вероятность)
            # обнаружение текущего объекта
            scores = detection[5:]
            #print(detection[0:4])
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            #print(confidence)
            # отбросьте слабые прогнозы, убедившись, что обнаруженные
            # вероятность больше минимальной вероятности
            if confidence >= CONFIDENCE:
                #print("CONFIRMED")
                # масштабируем координаты ограничивающего прямоугольника относительно
                # размер изображения, учитывая, что YOLO на самом деле
                # возвращает центральные координаты (x, y) ограничивающего
                # поля, за которым следуют ширина и высота поля
                box = detection[:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                # используем центральные координаты (x, y) для получения вершины и
                # и левый угол ограничительной рамки
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # обновить наш список координат ограничивающего прямоугольника, достоверности,
                # и идентификаторы класса
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    #print(detection.shape)

    # выполнить не максимальное подавление с учетом оценок, определенных ранее
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, IOU_THRESHOLD)
    #print(len(boxes))
    # убедитесь, что обнаружен хотя бы один объект
    if len(idxs) > 0:
        # перебираем сохраняемые индексы
        for i in idxs.flatten():
            # извлекаем координаты ограничивающего прямоугольника
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]
            # рисуем прямоугольник ограничивающей рамки и подписываем на изображении
            color = [int(c) for c in colors[class_ids[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color=color, thickness=thickness)
            text = f"{labels[class_ids[i]]}: {confidences[i]:.2f}"
            # вычисляем ширину и высоту текста, чтобы определить прозрачные поля в качестве фона текста
            (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]
            text_offset_x = x
            text_offset_y = y - 5
            box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
            overlay = image.copy()
            cv2.rectangle(overlay, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED)
            # добавить непрозрачность (прозрачность поля)
            image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
            # теперь поместим текст (метка: доверие%)
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=font_scale, color=(0, 0, 0), thickness=thickness)



#cv2.imwrite(filename + "_yolo3." + ext, image)
#while True:
    cv2.imshow("WIN", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()