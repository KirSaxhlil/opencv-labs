import cv2 as cv
import numpy as np

video = cv.VideoCapture(0)
#ok, img = video.read()
#w = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
#h = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
settings_window_name = "Settings Window"
cv.namedWindow(settings_window_name)

LowH = 0
HighH = 255
LowS = 175
HighS = 255
LowV = 100
HighV = 255
def on_LowH_trackback(val):
    global LowH
    LowH = val
def on_HighH_trackback(val):
    global HighH
    HighH = val
def on_LowS_trackback(val):
    global LowS
    LowS = val
def on_HighS_trackback(val):
    global HighS
    HighS = val
def on_LowV_trackback(val):
    global LowV
    LowV = val
def on_HighV_trackback(val):
    global HighV
    HighV = val

cv.createTrackbar("Low H", settings_window_name, 165, 180, on_LowH_trackback)
cv.createTrackbar("Max H", settings_window_name, 15, 180, on_HighH_trackback)
cv.createTrackbar("Low S", settings_window_name, 150, 255, on_LowS_trackback)
cv.createTrackbar("Max S", settings_window_name, 255, 255, on_HighS_trackback)
cv.createTrackbar("Low V", settings_window_name, 85, 255, on_LowV_trackback)
cv.createTrackbar("Max V", settings_window_name, 255, 255, on_HighV_trackback)

while True:
    ok, frame = video.read()
    if not ok:
        break
    frame = cv.flip(frame, 1)
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    if LowH < HighH:
        mask = cv.inRange(hsv, (LowH, LowS, LowV), (HighH, HighS, HighV))
        #res = cv.bitwise_and(frame, frame, mask=mask)
    else:
        mask1 = cv.inRange(hsv, (0, LowS, LowV), (HighH, HighS, HighV))
        mask2 = cv.inRange(hsv, (LowH, LowS, LowV), (360, HighS, HighV))
        mask = cv.bitwise_or(mask1, mask2)
        #res1 = cv.bitwise_and(frame, frame, mask=mask1)
        #res2 = cv.bitwise_and(frame, frame, mask=mask2)
        #res = cv.bitwise_or(res1, res2)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv.erode(mask, kernel, iterations=1)
    mask = cv.dilate(mask, kernel, iterations = 1)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    moments = cv.moments(mask, True)
    if moments['m00'] > 100:
        X = int(moments['m10']/moments['m00'])
        Y = int(moments['m01']/moments['m00'])

        U20 = int(moments['m20']-X*moments['m10'])
        XX = int(U20/moments['m11'])

        #print(XX)
        #cv.rectangle(frame, (X-100, Y-100), (X+100, Y+100), (0,0,0), 3)
        cv.circle(frame, (X, Y), 100, (0,0,0), 3)
        cv.line(frame, (X, Y+150), (X, Y-150), (0,0,0), 3)
        cv.line(frame, (X + 150, Y), (X - 150, Y), (0, 0, 0), 3)
        #print(moments['m01'])

    res = cv.bitwise_and(frame, frame, mask=mask)
    cv.imshow("WINDAW", frame)

    key = cv.waitKey(1)
    if key == ord('q') or key == 27:
        break

cv.destroyAllWindows()