import preprocessing

import cv2
import numpy as np

import pytesseract
from pytesseract import Output
pytesseract.pytesseract.tesseract_cmd = r'E:\Tesseract-OCR\tesseract'
custom_oem_psm_config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ --oem 3 --psm 10'

from keras.models import load_model
model = load_model('./model/model_print400.h5')
word_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',
             13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X', 24:'Y',25:'Z'}


def evaluate_char_model(img_char):
    img_resize = cv2.resize(img_char, (28, 28))
    """cv2.imshow("1", img_resize)
    cv2.waitKey()
    cv2.destroyAllWindows()"""
    img_char = preprocessing.adaptative_threshold(img_resize)
    img_final = np.reshape(img_char, (1, 28, 28, 1))
    char = word_dict[np.argmax(model.predict(img_final))]
    #print(char)
    return char

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
        cv2.imwrite('./data/' + str(i+700) + '.png', img_char)

def evaluate_model(pred, gt):
    acc = sum(pred == gt) / pred.size
    print("Accuracy: ", acc)
    return acc