import math
import numpy as np

import cv2 as cv

kernel_size = 5
sigma = 1

def gauss(x, y):
    mw = int(kernel_size / 2)
    #mw = kernel_size / 2
    return pow(1/(2*math.pi*pow(sigma, 2)), -(pow(x-mw, 2) + pow(y-mw, 2))/(2*pow(sigma, 2)))

def make_kernel():
    kernel = []
    for i in range(kernel_size):
        row = []
        for j in range(kernel_size):
            row.append(gauss(i,j))
        kernel.append(row)
    normalize(kernel)
    return kernel

def make_kernel2():
    kernel = np.full((kernel_size, kernel_size), 0)
    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel[i, j] = gauss(i,j)
    return normalize(kernel)
def normalize(matrix):
    mat_sum = matrix.sum()
    return matrix * (1/mat_sum)

def blur(img, gauss):
    new_img = img.copy()
    height, width = img.shape
    ker_bound = math.floor(kernel_size/2)
    for x in range(ker_bound, width-ker_bound):
        for y in range(ker_bound, height-ker_bound):
            new_img[y,x] = np.multiply(img[y-ker_bound:y+ker_bound+1, x-ker_bound:x+ker_bound+1], gauss).sum()
    return new_img


file_name1 = "LuminescentCore_Camera_Point_002.png"
file_name2 = "quality.jpg"
file_name3 = "ScreenShot104.bmp"
image = cv.imread(file_name2)


gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

new_image = blur(gray, make_kernel2())
cv.imshow("ORIGINAL", gray)
cv.imshow("MY BLUR", new_image)
cvblured = cv.blur(gray, (kernel_size,kernel_size), sigma)
cv.imshow("CV BLUR", cvblured)
cv.waitKey(0)
cv.destroyAllWindows()