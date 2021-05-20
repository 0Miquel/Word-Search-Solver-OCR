import preprocessing

import cv2
import numpy as np

import pytesseract
from pytesseract import Output
pytesseract.pytesseract.tesseract_cmd = r'E:\Tesseract-OCR\tesseract'
custom_oem_psm_config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ --oem 3 --psm 10'

from keras.models import load_model
model = load_model('./model/model_print.h5')
word_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',
             13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X', 24:'Y',25:'Z'}


def evaluate_char_model(img_char):
    img_resize = cv2.resize(img_char, (28, 28))
    """cv2.imshow("1", img_resize)
    cv2.waitKey()
    cv2.destroyAllWindows()"""
    _, img_thresh = cv2.threshold(img_resize, 100, 255, cv2.THRESH_BINARY_INV)
    img_final = np.reshape(img_thresh, (1, 28, 28, 1))
    char = word_dict[np.argmax(model.predict(img_final))]
    #print(char)
    return char


"""
['O' 'M' 'Y' 'P' 'A' 'P' 'O' 'A' 'T' 'A' 'L' 'C' 'A' 'P' 'T' 'S' 'C' 'A'
 'A' 'F' 'G' 'O' 'D' 'A' 'G' 'A' 'P' 'Y' 'D' 'F' 'S' 'P' 'E' 'C' 'H' 'O'
 'M' 'L' 'A' 'P' 'E' 'O' 'T' 'A' 'E' 'P' 'X' 'E' 'T' 'A' 'E' 'C' 'A' 'E'
 'J' 'E' 'U' 'A' 'L' 'A' 'N' 'A' 'A' 'E' 'F' 'L' 'O' 'G' 'A' 'T' 'A' 'E'
 'K' 'O' 'J' 'S' 'S' 'A' 'F' 'E' 'E' 'E' 'Q' 'L' 'P' 'B' 'A' 'A' 'C' 'M'
 'Y' 'N' 'C' 'H' 'E' 'A' 'M' 'P' 'J' 'M' 'T' 'A' 'L' 'A' 'L' 'Q' 'A' 'S'
 'E' 'A' 'B' 'R' 'S' 'T' 'A' 'A' 'E' 'A' 'A' 'M' 'T' 'E' 'A' 'C' 'O' 'T'
 'S' 'E' 'A' 'P' 'A' 'A' 'E' 'U' 'F' 'F' 'A' 'E' 'O' 'A' 'S' 'M' 'A' 'O'
 'D' 'A' 'S' 'F' 'O' 'Z' 'R' 'M' 'T' 'P' 'P' 'S' 'A' 'B' 'O' 'A' 'T' 'E'
 'T' 'E' 'M' 'C' 'A' 'O' 'C' 'T' 'T' 'A' 'A' 'T' 'A' 'A' 'T' 'A' 'A' 'C'
 'A' 'T' 'O' 'A' 'T' 'N' 'E' 'D' 'N' 'O' 'L' 'M' 'M' 'F' 'A' 'F' 'T' 'N'
 'A' 'Z' 'P' 'A' 'S' 'M' 'A' 'A' 'A' 'E' 'O' 'E' 'E' 'E' 'T' 'E' 'C' 'L'
 'E' 'A' 'O' 'A' 'E' 'L' 'O' 'G' 'T' 'O' 'P' 'S' 'R' 'C' 'N' 'N' 'L' 'A'
 'U' 'U' 'E' 'T' 'C' 'P' 'A' 'L' 'L' 'Y' 'M' 'M' 'S' 'P' 'A' 'T' 'U' 'O'
 'E' 'C' 'C' 'T' 'R' 'T' 'A' 'A' 'A' 'A' 'L' 'C' 'N' 'Y' 'N' 'E' 'O' 'L'
 'S' 'P' 'O' 'M' 'A' 'L' 'T' 'S' 'S' 'L' 'C' 'A' 'M' 'P' 'A' 'A' 'O' 'N'
 'B' 'O' 'T' 'C' 'E' 'A' 'T' 'D' 'C' 'L' 'O' 'E']
"""

def evaluate_char_tesseract(img_char):
    ocr = pytesseract.image_to_data(img_char, output_type=Output.DICT, config=custom_oem_psm_config)
    #print(ocr['text'][-1], ocr['conf'][-1])
    #cv2.imshow("1", img_char)
    #cv2.waitKey()
    #cv2.destroyAllWindows()
    char = ocr['text'][-1]
    return char

def predict_chars(inv_thresh, thresh):
    result = []
    ctrs = preprocessing.get_contours(inv_thresh)
    for i, ctr in enumerate(ctrs):
        # Get bounding box
        x, y, w, h = ctr
        # Getting character
        img_char = thresh[y-5:y + h+5, x-5:x + w+5]
        # evaluate character
        char = evaluate_char_model(img_char)
        result.append(char)

    result = np.array(result)
    return result

def generate_dataset(inv_thresh, thresh):
    ctrs = preprocessing.get_contours(inv_thresh)
    for i, ctr in enumerate(ctrs):
        # Get bounding box
        x, y, w, h = ctr
        # Getting character
        img_char = thresh[y - 5:y + h + 5, x - 5:x + w + 5]
        #cv2.imshow("1",img_char)
        # save character
        img_resize = cv2.resize(img_char, (28, 28))
        img_char = preprocessing.adaptative_threshold(img_resize)
        """cv2.imshow("2", img_char)
        cv2.waitKey()
        cv2.destroyAllWindows()"""
        cv2.imwrite('./data/' + str(i) + '.png', img_char)