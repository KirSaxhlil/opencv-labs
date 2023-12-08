from os import listdir
from os.path import isfile, join

import pytesseract as pt
import cv2 as cv

pt.pytesseract.tesseract_cmd = (r"C:\Program Files\Tesseract-OCR\tesseract.exe")

senya_img_src = "senya.jpg"
todd_img_src = "todd.jpg"
ena_img_src = "ena.jpg"
sringe_img_src = "sringe.jpg"
design_img_src = "design.jpg"
rtx_img_src = "rtx.jpg"
doom_img_src = "doom.jpg"

ing = cv.imread(doom_img_src)
text = pt.image_to_string(ing, lang="rus")
print("TEXT: "+text)
#cv.imshow("WINDAW", ing)
#print(ing)



texts = open("texts.txt","r").read().splitlines()
print(texts)
#file1 =
#file1.close()

img_src = doom_img_src

capchi = []
capchi_path = "capchi/"
capchi = [f for f in listdir(capchi_path) if isfile(join(capchi_path, f))]
print(capchi)

#img = cv.imread(r"capchi\03 застыло безопасно.jpg")#(capchi_path + capchi[1])
#cv.imshow("WINDAW", img)
print(pt.get_languages())
def test_recognition(rec_type = None, val_type = None):
    global capchi
    global capchi_path
    for capcha in capchi:
        img = cv.imread(join(capchi_path, capcha))
        #cv.imshow("WINDAW", img)

        #text = pt.image_to_string(img, lang="rus+eng")
        #print(text + " : " + capcha[3:len(capcha)-1-4])

#test_recognition()

cv.waitKey(0)
cv.destroyAllWindows()