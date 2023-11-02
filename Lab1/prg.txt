import cv2
number_video = 0

def write_video_move(kernel_size, standard_deviation, thresh_value, min_area):
    global number_video
    number_video+=1
    video = cv2.VideoCapture(0, cv2.CAP_ANY)

    ret, frame = video.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (kernel_size, kernel_size), standard_deviation)

    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(r'.\output' + str(number_video) + '.mp4', fourcc, 20, (w, h))


    while True:
        # предыдущее изображение
        last_img = img.copy()
        ret, frame = video.read()
        if not ret:
            break
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), standard_deviation)

        # вычисляем разницу
        diff = cv2.absdiff(img, last_img)

        # бинаризируем её превращая пиксели, превышающие порог delta_tresh, в белый цвет, а остальные в черный
        #treshhold возвращает изображение и пороговое значение
        # сохраняем матрицу
        thresh = cv2.threshold(diff, thresh_value, 255, cv2.THRESH_BINARY)[1]

        # нахождение контуров - находим только внейшние контуры - сжимает горизонтальные, вертикальные и диагональные сегменты и оставляет только их конечные точки
        (conturs, _) = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # если на кадре есть хотя бы один контур, чья площадь достаточно большая то записываем кадр
        contur_areas = map(cv2.contourArea, conturs)
        #контуры с площадью больше заданной
        for area in contur_areas:
            if area>=min_area:
                video_writer.write(frame)
                break
    cv2.waitKey(0)
    video_writer.release()

kernel_size = 3
standard_deviation = 50
thresh_value = 60
min_area = 20
write_video_move(kernel_size, standard_deviation, thresh_value, min_area)