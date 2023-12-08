import cv2

face_cascade = cv2.CascadeClassifier('cascad_trained/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while True:
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

cap.release()
cv2.destroyAllWindows()