import cv2
import numpy as np
import imutils
from skimage import measure

def get_contours(inv_thresh):
    ctrs, hier = cv2.findContours(inv_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_ctrs = sort_ctrs(ctrs)
    return sorted_ctrs

def sort_ctrs(ctrs):
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

def draw_ctrs(inv_thresh, thresh):
    ctrs = get_contours(inv_thresh)
    aux_thresh = thresh.copy()
    for i, ctr in enumerate(ctrs):
        # Get bounding box
        x, y, w, h = ctr
        cv2.rectangle(aux_thresh, (x-5, y-5), (x + w+5, y + h+5), (90, 0, 255), 2)
    cv2.imshow('marked areas', aux_thresh)
    cv2.waitKey()
    cv2.destroyAllWindows()

def read_gray_image(img_path):
    # read image
    img = cv2.imread(img_path)
    # grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def binarize_image(gray):
    # binarize
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return thresh

def inv_image(thresh):
    inv_thresh = cv2.bitwise_not(thresh)
    return inv_thresh

def adaptative_threshold(gray):
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                cv2.THRESH_BINARY, 27, 18)
    return thresh

def remove_extra_information(thresh, inv_thresh):
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

    n_labels, labels = cv2.connectedComponents(new_box)
    new_thresh = thresh.copy()
    # erase the content of the thresholded image that is outside of the box
    for i in range(len(box)):
        for j in range(len(box[i])):
            if labels[i][j] != 2:
            #if labels[i][j] == 1:
                new_thresh[i][j] = 255

    return new_thresh, box

def remove_isolated_pixels(thresh):
    inv_thresh = inv_image(thresh)
    connectivity = 8
    output  = cv2.connectedComponentsWithStats(inv_thresh, connectivity, cv2.CV_32S)
    n_labels = output[0]
    labels = output[1]
    stats = output[2]

    for label in range(n_labels):
        if stats[label, cv2.CC_STAT_AREA] < 15:
            inv_thresh[labels == label] = 0

    new_thresh = inv_image(inv_thresh)

    return new_thresh, inv_thresh

def get_corners(box):
    gray = cv2.GaussianBlur(box, (5, 5), 0)
    edges = cv2.Canny(gray, 75, 200);

    cnts = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break

    screenCnt = np.array([screenCnt[0][0],screenCnt[3][0],screenCnt[2][0],screenCnt[1][0]], dtype="float32")
    rect = sort_corners(screenCnt)
    return rect

def sort_corners(corners):
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = corners.sum(axis=1)
    rect[0] = corners[np.argmin(s)]
    rect[2] = corners[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(corners, axis=1)
    rect[1] = corners[np.argmin(diff)]
    rect[3] = corners[np.argmax(diff)]

    return rect

def homography(corners, image):
    (tl, tr, br, bl) = corners

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")

    M, status = cv2.findHomography(corners, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped