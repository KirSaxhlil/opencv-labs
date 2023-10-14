import cv2 as cv

def print_image(index):
    image = None
    match index:
        case 1: image = cv.imread("LuminescentCore_Camera_Point_002.png")
        case 2: image = cv.imread("quality.jpg", cv.IMREAD_REDUCED_GRAYSCALE_8)
        case 3: image = cv.imread("ScreenShot104.bmp", cv.IMREAD_REDUCED_COLOR_8)
        case _: return
    cv.imshow("Windaw", image)
    cv.waitKey(0)

def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))

VIDEO_SPECIAL = 1
VIDEO_WRITE = 2
VIDEO_GRAYCAM = 4
VIDEO_RECTANGLE = 8
VIDEO_RECTANGLE_FILLED = 16
def video_processing(source, flags = 0):
    video = cv.VideoCapture(source)
    ok, img = video.read()
    w = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
    video_writer = None
    if flags & VIDEO_WRITE:
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        video_writer = cv.VideoWriter("output.mov", fourcc, 25, (w,h))
    while True:
        ok, img = video.read()
        if not ok:
            break
        if flags & VIDEO_SPECIAL:
            img = cv.flip(img, 1)
            img = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
        if flags & VIDEO_GRAYCAM:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            img = cv.flip(img, 1)
        if flags & VIDEO_RECTANGLE or flags & VIDEO_RECTANGLE_FILLED:
            color = (0, 0, 255)
            thinkness = 2
            if flags & VIDEO_RECTANGLE_FILLED:
                tmp_img = img
                if flags & VIDEO_SPECIAL:
                    tmp_img = cv.cvtColor(img, cv.COLOR_YCrCb2BGR)
                if flags & VIDEO_GRAYCAM:
                    tmp_img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
                tmp_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
                central = [0, 0, 0]
                central[0], central[1], central[2] = tmp_img[int(w/2), int(h/2)]
                print(central)
                #need = central.index(max(central))
                color = (0,0,0)
                if central[0] < 30 or central[0] > 150: #16.6 83.3
                    color = (0,0,255)
                elif central[0] >= 30 and central[0] < 90: #16.6 50
                    color = (0,255,0)
                else:
                    color = (255,0,0)
                #color = [0, 0, 0]
                #color[need] = 255
                thinkness = -1
            max_len = 200
            min_len = 50
            max_wid = 50
            min_wid = 5
            img = cv.rectangle(img, (int(w/2)-clamp(int(w/6), min_len, max_len), int(h/2)-clamp(int(h/20), min_wid, max_wid)), (int(w/2)+clamp(int(w/6), min_len, max_len),int(h/2)+clamp(int(h/20), min_wid, max_wid)), color, thinkness)
            img = cv.rectangle(img, (int(w / 2) - clamp(int(w/20), min_wid, max_wid), int(h / 2) - clamp(int(h/6), min_len, max_len)), (int(w / 2) + clamp(int(w/20), min_wid, max_wid), int(h / 2) + clamp(int(h/6), min_len, max_len)), color, thinkness)
        cv.imshow('Windaw', img)
        if flags & VIDEO_WRITE:
            video_writer.write(img)
        if cv.waitKey(1) == ord('q'):
            break
    video.release()

def RGB_to_HSV():
    frame = cv.imread("LuminescentCore_Camera_Point_002.png")
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    cv.imshow("Windaw2", hsv)
    cv.waitKey(0)
    cv.destroyWindow("Windaw2")

cv.namedWindow("Windaw", cv.WINDOW_NORMAL)

#print_image(1)
#RGB_to_HSV()
#print_image(2)
#print_image(3)

path_to_video = "SpaceExterScene_(3) (video-converter.com)_2.mp4"
path_to_phone_cam = "https:192.168.43.1:8080/video"

#video_processing(path_to_video)
#video_processing(path_to_video, VIDEO_SPECIAL)
#video_processing(path_to_video, VIDEO_WRITE)
#video_processing(0, VIDEO_GRAYCAM)
#video_processing(0, VIDEO_RECTANGLE)
#video_processing(0, VIDEO_WRITE)
video_processing(0, VIDEO_RECTANGLE_FILLED)
#video_processing(path_to_phone_cam)

cv.destroyAllWindows()