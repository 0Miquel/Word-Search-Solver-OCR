import cv2
import pytesseract
import numpy as np
import pandas as pd
import os
import time
from keras.models import load_model
from pytesseract import Output
from skimage import measure

pytesseract.pytesseract.tesseract_cmd = r'E:\Tesseract-OCR\tesseract'
custom_oem_psm_config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ --oem 3 --psm 10'
word_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X', 24:'Y',25:'Z'}
model = load_model('./model/model_hand.h5')

def find_contours(img, inv_thresh, thresh):
    """

    :param img:
    :param inv_thresh:
    :param thresh:
    :return:
    """
    result = []
    # find contours
    ctrs, hier = cv2.findContours(inv_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # sort contours
    sorted_ctrs = sort_ctrs(ctrs)
    for i, ctr in enumerate(sorted_ctrs):
        # Get bounding box
        x, y, w, h = ctr
        #x, y, w, h = cv2.boundingRect(ctr)
        # Getting character
        img_char = thresh[y-5:y + h+5, x-5:x + w+5]
        # evaluate character
        char = evaluate_char_model(img_char)
        result.append(char)
        # save character
        #cv2.imwrite('./data/' + str(i) + '.png', img_char)
        #cv2.rectangle(img, (x-5, y-5), (x + w+5, y + h+5), (90, 0, 255), 2)
    #cv2.imshow('marked areas', img)
    #cv2.waitKey()
    #cv2.destroyAllWindows()

    result = np.array(result)
    return result

def sort_ctrs(ctrs):
    """

    :param ctrs:
    :return:
    """
    bboxes = [cv2.boundingRect(i) for i in ctrs]
    bboxes = sorted(bboxes, key=lambda ctr: ctr[1])
    sorted_ctrs = []
    ctrs_i = []
    ctr_y = bboxes[0][1]
    for ctr in bboxes:
        if ctr[1] < ctr_y + 5 and ctr[1] > ctr_y - 5:
            ctrs_i.append(ctr)
            ctr_y = ctr[1]
        else:
            ctrs_i = sorted(ctrs_i, key=lambda ctr: ctr[0])
            sorted_ctrs = sorted_ctrs + ctrs_i
            ctrs_i = [ctr]
            ctr_y = ctr[1]
    ctrs_i = sorted(ctrs_i, key=lambda ctr: ctr[0])
    new_ctrs = sorted_ctrs + ctrs_i
    return new_ctrs

def evaluate_char_tesseract(img_char):
    """

    :param img_char:
    :return:
    """
    ocr = pytesseract.image_to_data(img_char, output_type=Output.DICT, config=custom_oem_psm_config)
    #print(ocr['text'][-1], ocr['conf'][-1])
    #cv2.imshow("1", img_char)
    #cv2.waitKey()
    #cv2.destroyAllWindows()
    char = ocr['text'][-1]
    return char

def evaluate_char_model(img_char):
    """

    :param img_char:
    :return:
    """
    img_resize = cv2.resize(img_char, (28, 28))
    """cv2.imshow("1", img_resize)
    cv2.waitKey()
    cv2.destroyAllWindows()"""
    _, img_thresh = cv2.threshold(img_resize, 100, 255, cv2.THRESH_BINARY_INV)
    img_final = np.reshape(img_thresh, (1, 28, 28, 1))
    char = word_dict[np.argmax(model.predict(img_final))]

    return char

def read_image(img_path):
    """

    :param img_path:
    :return:
    """
    # read image
    img = cv2.imread(img_path)
    return img

def get_gray_image(img):
    """

    :param img:
    :return:
    """
    # grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def binarize_image(gray):
    """

    :param gray:
    :return:
    """
    # binarize
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return thresh

def inv_image(thresh):
    """

    :param thresh:
    :return:
    """
    inv_thresh = cv2.bitwise_not(thresh)
    return inv_thresh

def adaptative_threshold(gray):
    """

    :param gray:
    :return:
    """
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                cv2.THRESH_BINARY, 27, 18)
    return thresh

def remove_extra_information(thresh, inv_thresh):
    """

    :param thresh:
    :param inv_thresh:
    :return:
    """
    # Get regions and get the maximum size region
    n_labels, labels = cv2.connectedComponents(inv_thresh)
    regions = measure.regionprops(labels)
    max_reg = max(regions, key=lambda item: item.area)
    max_label = max_reg.label

    # Get word search box
    box = labels
    box[box != max_label] = 255
    box[box == max_label] = 0
    box = np.uint8(box)

    # erode so as to thick the box and extract the different regions
    kernel = np.ones((3, 3), np.uint8)
    new_box = cv2.erode(box, kernel)
    # cv2.imshow('Box',new_box)

    n_labels, labels = cv2.connectedComponents(new_box)
    new_thresh = thresh.copy()
    # erase the content of the thresholded image that is outside of the box
    for i in range(len(box)):
        for j in range(len(box[i])):
            if labels[i][j] != 2:
                new_thresh[i][j] = 255

    return new_thresh

def remove_isolated_pixels(thresh):
    """

    :param thresh:
    :return:
    """
    inv_thresh = inv_image(thresh)
    connectivity = 8
    output  = cv2.connectedComponentsWithStats(inv_thresh, connectivity, cv2.CV_32S)
    n_labels = output[0]
    labels = output[1]
    stats = output[2]

    for label in range(n_labels):
        if stats[label, cv2.CC_STAT_AREA] < 5:
            inv_thresh[labels == label] = 0

    new_thresh = inv_image(inv_thresh)
    return new_thresh, inv_thresh


if __name__ == "__main__":
    mode = "hard"
    if mode == "easy":
        img = read_image('../images/mini_word.png')
        gray = get_gray_image(img)
        thresh = binarize_image(gray)
        inv_thresh = inv_image(thresh)
        result = find_contours(img, inv_thresh, thresh)
        result = result.reshape((10, 10))
        print(result)
    elif mode == "hard":
        img = read_image('../images/example6.jpg')
        gray = get_gray_image(img)
        thresh = adaptative_threshold(gray)
        inv_thresh = inv_image(thresh)
        new_thresh = remove_extra_information(thresh, inv_thresh)

        new_thresh2, inv_thresh2 = remove_isolated_pixels(new_thresh)
        result = find_contours(img, inv_thresh2, new_thresh2)
        result = result.reshape((15, 20))
        print(result)
