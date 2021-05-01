import cv2
import numpy as np
from skimage import measure
import pytesseract
from pytesseract import Output

pytesseract.pytesseract.tesseract_cmd = r'E:\Tesseract-OCR\tesseract'

# Read Image
img = cv2.imread('../images/example6.jpg')
img_copy = img.copy()

# Gray image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold the image
th3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,27,18)
cv2.imshow('Threshold',th3)
th3_copy = th3.copy()

# Invert image to find regions
inv_th3 = cv2.bitwise_not(th3)
#cv2.imshow('Inv threshold',inv_th3)

# Get regions and get the maximum size region
n_labels, labels = cv2.connectedComponents(inv_th3)
regions = measure.regionprops(labels)
max_reg = max(regions, key=lambda item: item.area)
max_label = max_reg.label

# Get word search box
box = labels
box[box != max_label] = 255
box[box == max_label] = 0
box = np.uint8(box)

# erode so as to thick the box and extract the different regions
kernel = np.ones((3,3),np.uint8)
new_box = cv2.erode(box, kernel)
#cv2.imshow('Box',new_box)

n_labels, labels = cv2.connectedComponents(new_box)
# erase the content of the thresholded image that is outside of the box
for i in range(len(box)):
    for j in range(len(box[i])):
        if labels[i][j] != 2:
            th3[i][j] = 255
cv2.imshow('Final',th3)
th3_copy = th3.copy()

# detect characters
custom_oem_psm_config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ --oem 3 --psm 6'
ocr = pytesseract.image_to_data(th3_copy, output_type=Output.DICT, config=custom_oem_psm_config, lang='eng')
boxes = len(ocr['text'])
texts = []
for i in range(boxes):
    if (int(ocr['conf'][i]) != -1):
        (x, y, w, h) = (ocr['left'][i], ocr['top'][i], ocr['width'][i], ocr['height'][i])
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        print(ocr['text'][i], ocr['conf'][i])
        texts.append(ocr['text'][i])

cv2.imshow('Detection',img_copy)





"""# CONTOURS
contours, hierarchy = cv2.findContours(th3_copy,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
	x,y,w,h = cv2.boundingRect(cnt)
	#bound the images
	cv2.rectangle(th3_copy,(x,y),(x+w,y+h),(0,255,0),3)

cv2.imshow('Contours',th3_copy)

xi = 0
yi = 0
i=0
for cnt in contours:
	x,y,w,h = cv2.boundingRect(cnt)

	if (abs(x-xi) > 20 or abs(y-yi) > 20) and (w > 10 or h > 10):
		cv2.imwrite("./data/" + str(i) + ".jpg", th3[y-8:y+h+8,x-8:x+w+8])
		i=i+1

	xi = x
	yi = y"""


cv2.waitKey()
cv2.destroyAllWindows()