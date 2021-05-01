import cv2
import pytesseract
import numpy as np
import time
from pytesseract import Output
from skimage import measure

pytesseract.pytesseract.tesseract_cmd = r'E:\Tesseract-OCR\tesseract'

custom_oem_psm_config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ --oem 3 --psm 6'

im = cv2.imread('../images/example2.png',0)
im = cv2.adaptiveThreshold(im,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
				cv2.THRESH_BINARY,27,18)
im_copy = im.copy()
cv2.imshow("1", im)
results = pytesseract.image_to_boxes(im, output_type=Output.DICT, config=custom_oem_psm_config)

"""inv_im = cv2.bitwise_not(im)
n_labels, labels = cv2.connectedComponents(inv_im)"""

for i in range(len(results['char'])):
	x0 = results['left'][i]
	y0 = im.shape[0]-results['top'][i]
	x1 = results['right'][i]
	y1 = im.shape[0]-results['bottom'][i]
	im = cv2.rectangle(im, (x0, y0), (x1, y1), (0, 255, 0), 2)
	#cv2.imwrite("./data/" + str(i) + ".jpg", im_copy[y0:y1, x0:x1])
cv2.imshow("2", im)

words = np.array(results['char'])

print(len(words))
for i in range(0,420, 20):
	print(words[i:i+20])

cv2.waitKey()
cv2.destroyAllWindows()