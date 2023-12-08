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

"""

import math

import cv2 as cv
import numpy as np

import os

file_name1 = "LuminescentCore_Camera_Point_002_us.png"
file_name2 = "quality_2.jpg"
src1 = "1.jpg"
src2 = "2.jpg"
src3 = "3.jpg"
src4 = "4.jpg"
src5 = "5.jpg"
file_name = src5

sigma_hWind = 6
sigma_Tol = 50
extreme_HWind = 2
threshold = 250
nByte = 1


kernel_size = 10
sigma = 1

sobelX = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
sobelY = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]


def SigmaFilter(Image, hWind, Tol, nByte):
    Output = Image.copy()
    height = len(Image)
    width = len(Image[0])
    Color = [0 for i in range(nByte)]#[0,0,0]
    Sum = [0 for i in range(nByte)]#[0,0,0]
    NumSum = [0 for i in range(nByte)]#[0,0,0]
    MinDif = [0 for i in range(nByte)]#[0,0,0]
    MaxDif = [0 for i in range(nByte)]#[0, 0, 0]
    for y in range(height):
        #print(y)
        yStart = max(y - hWind, 0)
        yEnd = min(y + hWind, height -1)
        for x in range(width):
            xStart = max(x - hWind, 0)
            xEnd = min(x+hWind, width-1)
            #print(xEnd)
            #print(x)
            for c in range(nByte):
                Color[c] = Image[y][x][c]
                Sum[c] = 0
                NumSum[c] = 0
                MinDif[c] = max(0, Color[c]-Tol)
                MaxDif[c] = min(255, Color[c]+Tol)
            j = yStart
            while j <= yEnd:
                i = xStart
                while i <= xEnd:
                    for c in range(nByte):
                        if MaxDif[c] > Image[j][i][c] > MinDif[c]:
                            Sum[c] += Image[j][i][c]
                            NumSum[c] += 1
                    i+=1
                j+=1
            for c in range(nByte):
                if NumSum[c] > 0:
                    Output[y][x][c] = Sum[c]/NumSum[c]
                else:
                    Output[y][x][c] = Image[y][x][c]
    return Output

def maxC(Color):
    return max(Color[0]*0.527, Color[1], Color[2]*0.713)
def ExtremUni(Image, hWind, nByte):
    #Output = Image.copy()
    height = len(Image)
    width = len(Image[0])
    Output = np.zeros((height, width, nByte), np.uint8)

    CenterColor = [0 for i in range(nByte)]#[0, 0, 0]
    Color = [0 for i in range(nByte)]#[0, 0, 0]
    Color1 = [0 for i in range(nByte)]#[0, 0, 0]
    Color2 = [0 for i in range(nByte)]#[0, 0, 0]

    for y in range(height):
        for x in range(width):
            Color2 = Color1 = Color = CenterColor = Image[y][x]
            #for c in range(3):
            #    Color2[c] = Color1[c] = Color[c] = CenterColor[c] = Image[y][x][c]
            MinLight = 1000
            MaxLight = 0
            k = -hWind
            while k <= hWind:
                if y+k >= 0 and y+k < height:
                    j = -hWind
                    while j <= hWind:
                        if x+j >= 0 and x+j < width:
                            Color = Image[y + k][x + j]
                            #for c in range(3):
                            #    Color[c] = Image[y+k][x+j][c]
                            if nByte == 1:
                                Light = Color[0]
                            else:
                                Light = maxC(Color)#max(Color[2], Color[1], Color[0])
                            if Light < MinLight:
                                MinLight = Light
                                Color1 = Color
                            if Light > MaxLight:
                                MaxLight = Light
                                Color2 = Color
                        j+=1
                k+=1
            if nByte == 1:
                CenterLight = CenterColor[0]
            else:
                CenterLight = maxC(CenterColor)#max(CenterColor[2], CenterColor[1], CenterColor[0])
            if CenterLight - MinLight < MaxLight - CenterLight:
                Output[y][x] = Color1
            else:
                Output[y][x] = Color2
            #for c in range(3):
            #    if CenterLight - MinLight < MaxLight - CenterLight:
            #        Output[y][x][c] = Color1[c]
            #    else:
            #        Output[y][x][c] = Color2[c]
    return Output

def SignDIF(Color1, Color2):
    iColor1 = Color1[1] * (256 ** 2) + Color1[2] * 256 + Color1[0]
    iColor2 = Color2[1] * (256 ** 2) + Color2[2] * 256 + Color2[0]
    if iColor1 - iColor2 < 0:
        return -1
    else:
        return 1


def ToEdge_bruh(LUT, DIF, SAD):
    if LUT == 1:
        return True

def ToEdge(LUT, DIF, SAD):
    #if LUT == 2 or LUT == 3 and DIF[2]*DIF[1] > 0 and abs(DIF[2]) >= abs(DIF[1]) or LUT == 1 and DIF[2]*DIF[3] > 0 and abs(DIF[2]) > abs(DIF[3]) or DIF[2]*DIF[3] < 0 or LUT == 4 and DIF[2]*DIF[3] > 0 and DIF[2]*DIF[1] > 0 and abs(DIF[2]) > abs(DIF[3]) and abs(DIF[2]) > abs(DIF[1]) or DIF[2]*DIF[3] < 0 and DIF[2]*DIF[1] < 0 or DIF[2]*DIF[1] > 0 and SAD[2] >= SAD[1] and DIF[2]*DIF[3] < 0 or LUT == 5 and DIF[2]*DIF[1] > 0 and SAD[2] >= SAD[1] and DIF[2]*DIF[0] > 0 and SAD[2] >= SAD[0] or DIF[2]*DIF[1] < 0 and DIF[2]*DIF[0] > 0 and SAD[2] >= SAD[0] or DIF[2]*DIF[1] < 0 and DIF[2]*DIF[0] < 0 or DIF[2]*DIF[1] > 0 and SAD[2] >= SAD[1] and DIF[2]*DIF[0] > 0 or LUT == 6 and DIF[2]*DIF[3] > 0 and SAD[2] > SAD[3] and DIF[2]*DIF[4] > 0 or SAD[2] > SAD[4] or DIF[2]*DIF[3] < 0 or DIF[2]*DIF[3] > 0 and SAD[2] > SAD[3] and DIF[2]*DIF[4] < 0 or LUT == 7 and not (DIF[2]*DIF[1] >= 0 and SAD[2] < SAD[1] or DIF[2]*DIF[3] > 0 and SAD[2] <= SAD[3] or DIF[2]*DIF[3] > 0 and DIF[2]*DIF[4] > 0 and SAD[2] <= SAD[4]) or LUT == 8 and ((DIF[2]*DIF[3] < 0 or SAD[2] >= SAD[1] and (DIF[2]*DIF[0] < 0 or SAD[2] >= SAD[0])) and (DIF[2]*DIF[3] <0 or SAD[2] > SAD[3] and (DIF[2]*DIF[4] < 0 or SAD[2] > SAD[4]))) or LUT == 9 and not (DIF[2]*DIF[1] > 0 and DIF[2]*DIF[0] >0 and SAD[2]<SAD[0] or DIF[2]*DIF[1] >=0 and SAD[2] <SAD[1] or DIF[2]*DIF[3] >=0 and SAD[2] <= SAD[3]):
    #    return True
    #else:
    #    return False
    if (LUT == 2 or LUT == 3) and (DIF[2]*DIF[1] > 0 and abs(DIF[2]) >= abs(DIF[1])):
        return True
    #elif LUT == 1 and (DIF[2]*DIF[3] > 0 and abs(DIF[2]) > abs(DIF[3]) or DIF[2]*DIF[3] < 0):
    elif LUT == 1 and (
        (
            DIF[2] * DIF[3] > 0 and abs(DIF[2]) > abs(DIF[3])
        ) or (
            DIF[2] * DIF[3] < 0
        )
    ):
        return True
    #elif LUT == 4 and (DIF[2]*DIF[3] > 0 and DIF[2]*DIF[1] > 0 and abs(DIF[2]) > abs(DIF[3]) and abs(DIF[2]) > abs(DIF[1]) or DIF[2]*DIF[3] < 0 and DIF[2]*DIF[1] < 0 or DIF[2]*DIF[1] > 0 and SAD[2] >= SAD[1] and DIF[2]*DIF[3] < 0):
    elif LUT == 4 and (
        (
            DIF[2]*DIF[3] > 0 and DIF[2]*DIF[1] > 0 and abs(DIF[2]) > abs(DIF[3]) and abs(DIF[2]) >= abs(DIF[1])
        ) or (
            DIF[2]*DIF[1] < 0 and DIF[2]*DIF[3] < 0
        ) or (
            DIF[2]*DIF[1] > 0 and SAD[2] >= SAD[1] and DIF[2]*DIF[3] < 0
        )
    ):
        return True
    elif LUT == 5 and (
        (
            DIF[2]*DIF[1] > 0 and SAD[2] >= SAD[1] and DIF[2]*DIF[0] > 0 and SAD[2] >= SAD[0]
        ) or (
            DIF[2]*DIF[1] < 0 and DIF[2]*DIF[0] > 0 and SAD[2] >= SAD[0]
        ) or (
            DIF[2]*DIF[1] < 0 and DIF[2]*DIF[0] < 0
        ) or (
            DIF[2]*DIF[1] > 0 and SAD[2] >= SAD[1] and DIF[2]*DIF[0] > 0 and False
        )
    ):
        return True
    elif LUT == 6 and (
        (
            DIF[2]*DIF[3] > 0 and SAD[2] > SAD[3] and (DIF[2]*DIF[4] > 0 or SAD[2] > SAD[4])
        ) or (
            DIF[2]*DIF[3] < 0
        ) or (
            DIF[2]*DIF[3] > 0 and SAD[2] > SAD[3] and DIF[2]*DIF[4] < 0
        )
    ):
        return True
    elif LUT == 7 and not(
        (
            DIF[2]*DIF[1] >= 0 and SAD[2] < SAD[1]
        ) or (
            DIF[2]*DIF[3] > 0 and SAD[2] <= SAD[3]
        ) or (
            DIF[2]*DIF[3] > 0 and DIF[2]*DIF[4] > 0 and SAD[2] <= SAD[4]
        )
    ):
        return True
    elif LUT == 8 and (
        (
            (DIF[2]*DIF[3] < 0 or SAD[2] >= SAD[1]) and (DIF[2]*DIF[0] < 0 or SAD[2] >= SAD[0])
        ) and (
            (DIF[2]*DIF[3] <0 or SAD[2] > SAD[3]) and (DIF[2]*DIF[4] < 0 or SAD[2] > SAD[4])
        )
    ):
        return True
    elif LUT == 9 and not(
        (
            DIF[2]*DIF[1] > 0 and DIF[2]*DIF[0] >0 and SAD[2]<SAD[0]
        ) or (
            DIF[2]*DIF[1] >=0 and SAD[2] <SAD[1]
        ) or (
            DIF[2]*DIF[3] >=0 and SAD[2] <= SAD[3]
        )
    ):
        return True
    #else:
    #    return False

# x > 5 and x < 10 or x > 12 and x < 16:
# x = 7: 1 and 1 or 0 and 1 = 1
# x = 11: 1 and 0 or 0 and 1 = 0
# x = 15: 1 and 0 or 1 and 1 = 1

LUT = [0, 0, 0, 0, 2, 2, 3, 5, 0, 0, 0, 0, 1, 1, 4, 9, 0, 0, 0, 0, 2, 2, 3, 5, 0, 0, 0, 0, 6, 6, 7, 8]
init_frame = cv.imread(file_name)
print("Loaded image")
init_frame = cv.cvtColor(init_frame, cv.COLOR_BGR2GRAY)
init_frame = cv.cvtColor(init_frame, cv.COLOR_GRAY2BGR)
sigma_frame = SigmaFilter(init_frame, sigma_hWind, sigma_Tol, 3) # cv.blur(init_frame, (6,6))#
print("Sigma filtering complete")
frame = ExtremUni(sigma_frame, extreme_HWind, 3)

#frame = SigmaFilter(frame, 2, 150, 3)
print("Extreme value filter complete")
#borders = frame.copy()


cv.imshow("WIN", init_frame)
cv.imshow("WIN2", sigma_frame)
cv.imshow("WIN3", frame)

def ColorDifAbs(Colorp, Colorh):
    Dif = 0
    for c in range(3):
        Dif += abs(Colorp[c] - Colorh[c])
    return (Dif)/3

def ColorDifSign(Colorp, Colorh):
    Dif = 0
    for c in range(3):
        Dif += abs(Colorp[c] - Colorh[c])
    if maxC(Colorp)-maxC(Colorh) > 0:
        Sign = 1
    else:
        Sign = -1
    return (Sign*Dif)/3

def LabelCell(Image, threshold, nByte):
    height = len(Image) * 2
    width = len(Image[0]) * 2
    borders = np.zeros((height, width, 1), np.uint8)

    Lab = 255
    Colorh = [0 for i in range(nByte)]
    Colorp = [0 for i in range(nByte)]
    Colorv = [0 for i in range(nByte)]

    y = 1
    while y < height:
        x = 1
        while x < width:
            if x >= 3:
                yy = int(y/2)
                xxv = int((x-2)/2)
                xxp = int(x/2)
                Colorv = Image[yy][xxv]
                Colorp = Image[yy][xxp]
                if nByte == 3:
                    difV = ColorDifAbs(Colorp, Colorv)
                else:
                    difV = abs(Colorp[0] - Colorv[0])
                if difV < 0:
                    difV = -difV
                if difV > threshold:
                    borders[y][x-1] = Lab
                    borders[y-1][x-1] += 1
                    if y+1 < height:
                        borders[y+1][x-1] += 1
            if y >= 3:
                xx = int(x/2)
                yyh = int((y-2)/2)
                yyp = int(y/2)
                Colorh = Image[yyh][xx]
                Colorp = Image[yyp][xx]
                if nByte == 3:
                    difH = ColorDifAbs(Colorp, Colorh)
                else:
                    difH = abs(Colorp[0] - Colorh[0])
                if difH > threshold:
                    borders[y-1][x] = Lab
                    borders[y-1][x-1] += 1
                    if x+1 < width:
                        borders[y-1][x+1] += 1
            x+=2
        y+=2
    return borders

def LabelCellSign(Image, threshold, nByte):
    height = len(Image) * 2 +1
    width = len(Image[0]) * 2 +1
    borders = np.zeros((height, width, 1), np.uint8)

    Lab = 255
    LabAdd = 255
    Colorh = [0 for i in range(nByte)]
    Colorp = [0 for i in range(nByte)]
    Colorv = [0 for i in range(nByte)]

    maxDif = 0
    minDif = 0

    y=1
    while y < height:
        State = 0
        xopt = -1
        xStartP = xStartM = -1
        x = 3
        while x < width:
            yy = int(y/2)
            xxv = int((x-2)/2)
            xxp = int(x/2)
            Colorv = Image[yy][xxv]
            Colorp = Image[yy][xxp]
            if nByte == 3:
                difV = ColorDifSign(Colorp, Colorv)
            else:
                difV = Colorp[0] - Colorv[0]
            if difV > threshold:
                Inp = 1
            else:
                if difV > -threshold:
                    Inp = 0
                else:
                    Inp = -1
            Contr = State * 3 + Inp

            match(Contr):
                case 4:
                    if x > xStartP and difV > maxDif:
                        maxDif = difV
                        xopt = x
                case 3:
                    borders[y][xopt-1] = Lab
                    borders[y-1][xopt-1] += LabAdd
                    if y+1 < height:
                        borders[y+1][xopt-1] += LabAdd
                    State = 0
                case 2:
                    borders[y][xopt - 1] = Lab
                    borders[y - 1][xopt - 1] += LabAdd
                    if y + 1 < height:
                        borders[y + 1][xopt - 1] += LabAdd
                    minDif = difV
                    xopt = x
                    xStartM = x
                    State = -1
                case 1:
                    maxDif = difV
                    xopt = x
                    xStartP = x
                    State = 1
                case -1:
                    minDif = difV
                    xopt = x
                    xStartM = x
                    State = -1
                case -2:
                    borders[y][xopt - 1] = Lab
                    borders[y - 1][xopt - 1] += LabAdd
                    if y + 1 < height:
                        borders[y + 1][xopt - 1] += LabAdd
                    maxDif = difV
                    xopt = x
                    xStartP = x
                    State = 1
                case -3:
                    borders[y][xopt - 1] = Lab
                    borders[y - 1][xopt - 1] += LabAdd
                    if y + 1 < height:
                        borders[y + 1][xopt - 1] += LabAdd
                    State = 0
                case -4:
                    if x > xStartM and difV < minDif:
                        minDif = difV
                        xopt = x
            x += 2
        y += 2

    x = 1
    while x < width:
        State = 0
        yopt = -1
        yStartP = yStartM = -1
        minDif = 0
        y = 3
        while y < height:
            xx = int(x / 2)
            yyh = int((y - 2) / 2)
            yyp = int(y / 2)
            Colorh = Image[yyh][xx]
            Colorp = Image[yyp][xx]
            if nByte == 3:
                difH = ColorDifSign(Colorp, Colorh)
            else:
                difH = Colorp[0] - Colorh[0]
            if difH > threshold:
                Inp = 1
            else:
                if difH > -threshold:
                    Inp = 0
                else:
                    Inp = -1
            Contr = State * 3 + Inp

            match (Contr):
                case 4:
                    if y > yStartP and difH > maxDif:
                        maxDif = difH
                        yopt = y
                case 3:
                    borders[yopt - 1][x] = Lab
                    borders[yopt - 1][x - 1] += LabAdd
                    if x + 1 < width:
                        borders[yopt - 1][x + 1] += LabAdd
                    State = 0
                case 2:
                    borders[yopt - 1][x] = Lab
                    borders[yopt - 1][x - 1] += LabAdd
                    if x + 1 < width:
                        borders[yopt - 1][x + 1] += LabAdd
                    minDif = difH ###
                    yStartM = y ###
                    yopt = y
                    State = -1
                case 1:
                    maxDif = difH
                    yopt = y
                    yStartP = y
                    State = 1
                case -1:
                    minDif = difH
                    yopt = y
                    yStartM = y
                    State = -1
                case -2:
                    borders[yopt - 1][x] = Lab
                    borders[yopt - 1][x - 1] += LabAdd
                    if x + 1 < width:
                        borders[yopt - 1][x + 1] += LabAdd
                    maxDif = difH ###
                    yStartP = y ###
                    yopt = y
                    State = 1
                case -3:
                    borders[yopt - 1][x] = Lab
                    borders[yopt - 1][x - 1] += LabAdd
                    if x + 1 < width:
                        borders[yopt - 1][x + 1] += LabAdd
                    State = 0
                case -4:
                    if y > yStartM and difH < minDif:
                        minDif = difH
                        yopt = y
            y += 2
        x += 2

    return borders

Stats = [0 for i in range(10)]

y = 0
while y < height:
    SAD = [0, 0, 0, 0, 0]
    DIF = [0, 0, 0, 0, 0]
    x = -3
    while x < width - 1:
        x1 = x+4
        if x1+1 < width:
            SAD[0] = 0
            yy = int(y/2)
            xx1 = int((x1+1)/2)
            xx2 = int((x1 - 1)/2)
            print(str(frame[yy][xx2]) + " " + str(frame[yy][xx1]))
            #print(xx2)
            for c in range(3):
                SAD[0] += abs(frame[yy][xx1][c] - frame[yy][xx2][c])
            DIF[0] = SAD[0] * SignDIF(frame[yy][xx1], frame[yy][xx2])
        else:
            DIF[0] = SAD[0] = 0
        Code = 0
        for k in range(5):
            if SAD[k] > threshold:
                Code += 2 ** (k)
        print(SAD[0])
        #if SAD[0] > threshold:
        #    borders[y][x1] = 255
        Stats[LUT[Code]] +=1
        if ToEdge(LUT[Code], DIF, SAD) and x1 < width:
            borders[y][x1] = 255
            if y-1 > 0:
                borders[y-1][x1] += 100
            if y+1 < height:
                borders[y+1][x1] += 100
        for i in range(4):
            SAD[4-i] = SAD[4-i-1]
            DIF[4-1] = DIF[4-i-1]
        x += 2
    y+=2
# 0 1 2 3 4 5 6 7 8 9
# -2 +4 =2
# 0 +4 = 4

print(Stats)


x = 0
while x < width:
    SAD = [0, 0, 0, 0, 0]
    DIF = [0, 0, 0, 0, 0]
    y = -3
    while y < height - 1:
        y1 = y+4
        if y1 +1 < height:
            SAD[0] = 0
            xx = int(x/2)
            yy1 = int((y+1)/2)
            yy2 = int((y-1)/2)
            for c in range(3):
                SAD[0] += abs(frame[yy1][xx][c] - frame[yy2][xx][c])
            DIF[0] = SAD[0] * SignDIF(frame[yy1][xx], frame[yy2][xx])
        else:
            DIF[0] = SAD[0] = 0
        Code = 0
        for k in range(5):
            if SAD[k] > threshold:
                Code += 2 ** (k)
        #print(LUT[Code])
        if ToEdge(LUT[Code], DIF, SAD) and y1 < height:
            borders[y1][x] = 255
            borders[y1][x-1] += 100
            borders[y1][x+1] += 100
        for i in range(4):
            SAD[4 - i] = SAD[4 - i - 1]
            DIF[4 - 1] = DIF[4 - i - 1]
        y += 2
    x+=2

borders = SigmaFilter(borders, 2, 50, 1)
print("Sigma filtering complete")
borders = ExtremUni(borders, 1, 1)
print("Extreme value filter complete")

# print("Градиенты:")
# print(grads)
# print("Углы:")
# print(dirs)
#cv.imshow("WIN", frame)

borders = LabelCellSign(frame, threshold, 3)
new_file_name = [0,0,0,0]
for i in range(4):
    new_file_name[i] = file_name.split('.')[0] + f"_{i}_{sigma_hWind}_{sigma_Tol}_{extreme_HWind}_{threshold}_gray.png"
cv.imwrite(os.path.join("out", new_file_name[0]), init_frame)
cv.imwrite(os.path.join("out", new_file_name[1]), sigma_frame)
cv.imwrite(os.path.join("out", new_file_name[2]), frame)
cv.imwrite(os.path.join("out", new_file_name[3]), borders)
print(os.listdir())

cv.imshow("WINDAW", borders)
cv.waitKey(0)
cv.destroyAllWindows()
"""