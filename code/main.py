import preprocessing
import predictor
import cv2
import solver

mode = "predict"

if mode == "preprocessing":
    #gray image
    gray = preprocessing.read_gray_image('../images/image7.jpeg')
    #threshold
    thresh = preprocessing.adaptative_threshold(gray)
    cv2.imshow("First thresh",thresh)
    #inverse threhsold
    inv_thresh = preprocessing.inv_image(thresh)
    #remove extra info
    thresh, box = preprocessing.remove_extra_information(thresh, inv_thresh)
    cv2.imshow("Second thresh", thresh)
    #homography
    corners = preprocessing.get_corners(box)
    warped = preprocessing.homography(corners, thresh)
    cv2.imshow("Homography", warped)
    #remove isolated pixels
    thresh, inv_thresh = preprocessing.remove_isolated_pixels(warped)
    cv2.imshow("Third thresh", thresh)
    cv2.waitKey()
    cv2.destroyAllWindows()
    #draw characters contours
    preprocessing.draw_ctrs(inv_thresh, thresh)
elif mode == "predict":
    gray = preprocessing.read_gray_image('../images/image7.jpeg')
    # threshold
    thresh = preprocessing.adaptative_threshold(gray)
    # inverse threhsold
    inv_thresh = preprocessing.inv_image(thresh)
    # remove extra info
    thresh, box = preprocessing.remove_extra_information(thresh, inv_thresh)
    # homography
    corners = preprocessing.get_corners(box)
    warped = preprocessing.homography(corners, thresh)
    # remove isolated pixels
    thresh, inv_thresh = preprocessing.remove_isolated_pixels(warped)
    result = predictor.predict_chars(inv_thresh,thresh)
    print(result)
elif mode == "generate_dataset":
    gray = preprocessing.read_gray_image('../images/example2.png')
    # threshold
    thresh = preprocessing.adaptative_threshold(gray)
    # inverse threhsold
    inv_thresh = preprocessing.inv_image(thresh)
    #generate dataset
    predictor.generate_dataset(inv_thresh, thresh)


"""elif mode == "solve":
        matrix =[['s','d','o','g'],['z','u','c','a'],['a','a','x','t'],['t','e','t','k']]
        word = 'cat'
        positions, trobat = word_search_solver(matrix, word)
    elif mode == "test":
        img = read_image('../images/example6.jpg')
        img2 = read_image('../images/example41.jpeg')
        #img = get_gray_image(img)
        #img2 = get_gray_image(img2)
        im, h = alignImages(img, img2)
        plt.imshow(im)"""