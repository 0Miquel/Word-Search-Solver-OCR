import preprocessing
import predictor
import cv2
import solver
import ground_truth

mode = "predict"

# PREPROCESSING
img = preprocessing.read_image('../images/image4.jpeg')
gray = preprocessing.gray_image(img)
# threshold
thresh = preprocessing.adaptative_threshold(gray)
# inverse threhsold
inv_thresh = preprocessing.inv_image(thresh)
# remove extra info
thresh, box = preprocessing.remove_extra_information(thresh, inv_thresh)
# homography
corners = preprocessing.get_corners(box)
thresh_warped, img_warped = preprocessing.homography(corners, thresh, img)

# remove isolated pixels
thresh, inv_thresh = preprocessing.remove_isolated_pixels(thresh_warped)


if mode == "visualize":
    #draw characters contours
    preprocessing.draw_ctrs(inv_thresh, thresh)
elif mode == "predict":
    result, contours = predictor.predict_chars(inv_thresh,thresh)
    m_res = result.reshape((contours.shape[0],contours.shape[1]))
    print(m_res)
    #acc = predictor.evaluate_model(result, ground_truth.image8_gt)

    while True:
        word = input("Enter word: ")
        if word == "q":
            break
        positions, found = solver.word_search_solver(m_res, word, False)
        if found:
            print("Found")
            img_warped = preprocessing.draw_results(contours, positions, img_warped)
            preprocessing.undo_homography(corners, img, img_warped)
        else:
            print("Not found")

elif mode == "generate_dataset":
    #generate dataset
    predictor.generate_dataset(inv_thresh, thresh)
