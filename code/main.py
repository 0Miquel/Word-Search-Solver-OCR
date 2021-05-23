import preprocessing
import predictor
import cv2
import solver
import ground_truth

mode = "solve"

# PREPROCESSING
gray = preprocessing.read_gray_image('../images/image8.jpeg')
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
    print(result.reshape((15,20)))
    acc = predictor.evaluate_model(result, ground_truth.image8_gt)
elif mode == "generate_dataset":
    #generate dataset
    predictor.generate_dataset(inv_thresh, thresh)


elif mode == "solve":
    matrix =[['s','d','t','g'],['z','a','x','a'],['c','a','x','t'],['c','e','t','k']]
    word = 'cat'
    positions, trobat = solver.word_search_solver(matrix, word, False)
    
