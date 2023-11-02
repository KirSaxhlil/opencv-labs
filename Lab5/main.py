import cv2 as cv

kernel_size = 5
sigma = 10
min_area = 600
thresh_value = 15
path_to_video = "ЛР4_main_video.mov"
path_to_output = "output.mov"

def write_video_move(kernel_size, sigma, thresh_value, min_area, source):
    video = cv.VideoCapture(source, cv.CAP_ANY)

    ok, frame = video.read()
    img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    img = cv.GaussianBlur(img, (kernel_size, kernel_size), sigma)

    w = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    video_writer = cv.VideoWriter(path_to_output, fourcc, 30, (w, h))

    while True:
        last_img = img.copy()
        ok, frame = video.read()
        if not ok:
            break
        #cv.imshow('Windaw', frame)
        img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        img = cv.GaussianBlur(img, (kernel_size, kernel_size), sigma)

        diff = cv.absdiff(img, last_img)
        thresh = cv.threshold(diff, thresh_value, 255, cv.THRESH_BINARY)[1]
        (contours, _) = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contour_areas = map(cv.contourArea, contours)
        for area in contour_areas:
            if area>=min_area:
                video_writer.write(frame)
                break
    #cv.waitKey(0)
    video_writer.release()

#cv.namedWindow("Windaw", cv.WINDOW_NORMAL)
write_video_move(kernel_size, sigma, thresh_value, min_area, path_to_video)