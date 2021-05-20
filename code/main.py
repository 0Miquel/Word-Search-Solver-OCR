import preprocessing
import predictor
import cv2
import solver
import ground_truth

mode = "visualize"

# PREPROCESSING
gray = preprocessing.read_gray_image('../images/image6.jpg')
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


if mode == "visualize":
    #draw characters contours
    preprocessing.draw_ctrs(inv_thresh, thresh)
elif mode == "predict":
    result = predictor.predict_chars(inv_thresh,thresh)
    print(result)
    acc = predictor.evaluate_model(result, ground_truth.image3_gt)
elif mode == "generate_dataset":
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