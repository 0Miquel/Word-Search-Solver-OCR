import cv2
import numpy as np
from skimage import measure
import pytesseract
from pytesseract import Output

pytesseract.pytesseract.tesseract_cmd = r'E:\Tesseract-OCR\tesseract'

# Read Image
img = cv2.imread('../images/example2.png')

# Gray image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold the image
th3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,27,18)
cv2.imshow('Threshold',th3)

inv_th3 = cv2.bitwise_not(th3)
cv2.imshow('inv Threshold',inv_th3)
# detect characters
custom_oem_psm_config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ --oem 3 --psm 6'


ocr = pytesseract.image_to_data(th3, output_type=Output.DICT, config=custom_oem_psm_config, lang='eng')
boxes = len(ocr['text'])
texts = []
for i in range(boxes):
    if (int(ocr['conf'][i]) != -1):
        (x, y, w, h) = (ocr['left'][i], ocr['top'][i], ocr['width'][i], ocr['height'][i])
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        print(ocr['text'][i], ocr['conf'][i])

cv2.imshow('Detection',img)

cv2.waitKey()
cv2.destroyAllWindows()