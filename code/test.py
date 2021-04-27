import cv2
import pytesseract
import numpy as np

img = cv2.imread('../images/word-search-example.png')
result = pytesseract.image_to_string(img)
print(result)