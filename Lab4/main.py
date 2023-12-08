import math

import cv2 as cv

file_name1 = "LuminescentCore_Camera_Point_002_s.png"
file_name2 = "quality.jpg"
file_name = file_name1
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

frame = cv.imread(file_name)
gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
blured = cv.blur(gray, (kernel_size,kernel_size), sigma)
borders = blured.copy()

grads = [[0 for i in range(len(borders[0]))] for j in range(len(borders))]
dirs = [[0 for i in range(len(borders[0]))] for j in range(len(borders))] #[[0]*len(borders[0])]*len(borders)
max_grad = 0

#test = [[0 for i in range(3)] for j in range(3)]
#test[1][1] = 1
#print(test)

for y in range(len(grads)):
    for x in range(len(grads[0])):
        if x == 0 or x == len(borders[0])-1 or y == 0 or y == len(borders)-1:
            grads[y][x] = 0
            dirs[y][x] = -1
            continue
        img_part = blured[y - 1:y + 2, x - 1:x + 2]
        Gx = half_grad(img_part, sobelX)
        Gy = half_grad(img_part, sobelY)
        if Gx == 0 and Gy == 0:
            grads[y][x] = 0
            dirs[y][x] = -1
            continue
        tg = Gy / Gx
        dirs[y][x] = get_dir(Gx, Gy, tg)
        grads[y][x] = math.sqrt(Gx**2 + Gy**2)
        if grads[y][x] > max_grad:
            max_grad = grads[y][x]

for y in range(len(borders)):
    for x in range(len(borders[y])):
        if x == 0 or x == len(borders[y])-1 or y == 0 or y == len(borders)-1:
            borders[y][x] = 0
            continue

        if dirs[y][x] == 0 or dirs[y][x] == 4:
            if grads[y][x] > grads[y-1][x] and grads[y][x] > grads[y+1][x]:
                borders[y][x] = 255
            else:
                borders[y][x] = 0
        elif dirs[y][x] == 1 or dirs[y][x] == 5:
            if grads[y][x] > grads[y-1][x+1] and grads[y][x] > grads[y+1][x-1]:
                borders[y][x] = 255
            else:
                borders[y][x] = 0
        elif dirs[y][x] == 2 or dirs[y][x] == 6:
            if grads[y][x] > grads[y][x+1] and grads[y][x] > grads[y][x-1]:
                borders[y][x] = 255
            else:
                borders[y][x] = 0
        elif dirs[y][x] == 3 or dirs[y][x] == 7:
            if grads[y][x] > grads[y-1][x-1] and grads[y][x] > grads[y+1][x+1]:
                borders[y][x] = 255
            else:
                borders[y][x] = 0
        else:
            borders[y][x] = 0

low_level = max_grad//10
high_level = max_grad//5

for y in range(len(borders)):
    for x in range(len(borders[y])):
        if x == 0 or x == len(borders[y])-1 or y == 0 or y == len(borders)-1:
            continue
        if borders[y][x] == 255:
            if grads[y][x] < low_level:
                borders[y][x] = 0
            elif grads[y][x] > high_level:
                continue
            else:
                has_nbr = False
                for xx in range(-1, 2):
                    if has_nbr == True:
                        break
                    for yy in range(-1, 2):
                        if (borders[y+yy][x+xx] == 255):
                            has_nbr = True
                            break
                if has_nbr == True:
                    continue
                else:
                    borders[y][x] = 0

#print("Градиенты:")
#print(grads)
#print("Углы:")
#print(dirs)
cv.imshow("WIN", gray)
cv.imshow("WINDAW", borders)
cv.waitKey(0)
cv.destroyAllWindows()