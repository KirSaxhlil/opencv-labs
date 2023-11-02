import math

import cv2 as cv

file_name1 = "LuminescentCore_Camera_Point_002_s.png"
kernel_size = 10
sigma = 1

sobelX = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
sobelY = [[-1, -2, -1], [0, 0, 0], [1, 2 ,1]]

def half_grad(img_part, operator):
    sum = 0
    for y in range(len(img_part)):
        for x in range(len(img_part[y])):
            sum += img_part[y][x]*operator[y][x]
    return sum

def get_dir(x, y, tg):
    if (x>0 and y<0 and tg<-2.414) or (x<0 and y<0 and tg>2.414) or (x==0 and y<0): return 0
    elif x>0 and y<0 and tg<-0.414: return 1
    elif (x>0 and y<0 and tg>-0.414) or (x>0 and y>0 and tg<0.414) or (x>0 and y==0): return 2
    elif x>0 and y>0 and tg<2.414: return 3
    elif (x>0 and y>0 and tg>2.414) or (x<0 and y>0 and tg<-2.414) or (x==0 and y>0): return 4
    elif x<0 and y>0 and tg<-0.414: return 5
    elif (x<0 and y>0 and tg>-0.414) or (x<0 and y<0 and tg<0.414) or (x<0 and y==0): return 6
    elif x<0 and y<0 and tg<2.414: return 7

def grad_length(img, x, y):
    img_part = img[y-1:y+2, x-1:x+2]
    #print(img_part)
    return math.sqrt(half_grad(img_part, sobelX)**2 + half_grad(img_part, sobelY)**2)

frame = cv.imread(file_name1)
gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
blured = cv.blur(gray, (kernel_size,kernel_size), sigma)
borders = blured.copy()
for y in range(len(borders)):
    #print(y)
    for x in range(len(borders[y])):
        if x == 0 or x == len(borders[y])-1 or y == 0 or y == len(borders)-1:
            borders[y][x] = 0
            continue
        img_part = blured[y-1:y+2, x-1:x+2]

        Gx = half_grad(img_part, sobelX)
        Gy = half_grad(img_part, sobelY)
        #if Gx != 0:

        if Gx == 0 and Gy == 0:
            borders[y][x] = 0
            continue
        tg = Gy / Gx
        #else:
        #    tg = 0
        dir = get_dir(Gx, Gy, tg)
        #print(str(Gx) + " " + str(Gy) + " " + str(tg) + " = " + str(dir))
        current_grad = math.sqrt(Gx**2 + Gy**2)
        if dir == 0 or dir == 4:
            if current_grad > grad_length(blured,x, y-1) and current_grad > grad_length(blured,x, y+1):
                borders[y][x] = 255
            else:
                borders[y][x] = 0
        elif dir == 1 or dir == 5:
            if current_grad > grad_length(blured, x+1, y-1) and current_grad > grad_length(blured, x-1, y+1):
                borders[y][x] = 255
            else:
                borders[y][x] = 0
        elif dir == 2 or dir == 6:
            if current_grad > grad_length(blured, x+1, y) and current_grad > grad_length(blured, x-1, y):
                borders[y][x] = 255
            else:
                borders[y][x] = 0
        elif dir == 3 or dir == 7:
            if current_grad > grad_length(blured, x-1, y-1) and current_grad > grad_length(blured, x+1, y+1):
                borders[y][x] = 255
            else:
                borders[y][x] = 0

cv.imshow("WINDAW", borders)
cv.waitKey(0)
cv.destroyAllWindows()