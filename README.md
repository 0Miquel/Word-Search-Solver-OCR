# Word search solver with OCR
![resultatSopaLletres](https://user-images.githubusercontent.com/48658941/120107885-07b29800-c163-11eb-82f1-0aded77600d0.png)

### Computer vision project from Universitat Autònoma de Barcelona

#

To acomplish the desired results in our word search solver with have implemented different modules:
- Pre processing: filtering, homographies and character detection.
- Model fine-tuning: train a pre-trained model with MNIST dataset to adapt it to our problem.
- Inference and solving: use of simple backtracking algorithm for solving the word search once every character have been fully recognized.

A part from pre-trained models we also have tried the open-source OCR Tesseract, but we did not achieve good enough results to solve word searches consistently.

## Authors
- Miquel Romero Blanch
- Gerard Graugés Bellver
