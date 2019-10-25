# Filter Images Using Modern Face Recognition
Train deep learning based face recognition system with only 5-10 images and accurately filter out images of specific person from directory of images.
> Implemented using OpenCV and Tensorflow.

With the rise of deep learning, there are many state-of-the art deep learning models which can be used for face detection and recognition. But they need to be trained on thousands of training samples. Instead, this project attempts to build face recognition system with near accurcay as state-of-the-art models while training on much less data.  
How does it work? See [wiki](https://github.com/xanjay/filter_images_using_modern_face_recognition/wiki#how-does-it-work). 
# Installation
**Dependencies**: Python3, OpenCV, Tensorflow, Scikit-Learn, DLib  
Install the requirements with PIP and get started. (Better use [virtualenv](https://virtualenv.pypa.io/en/latest/))
```sh
pip install -r requirements.txt
```
# Usage
You can train this system to recognize different faces and filter out the images of desired person from the image directory. Although
this project focuses on image filtering, there are many more use-cases of face recognition. You can inherit the face recognition system used here
and apply to your own application. 
## Training on your own dataset
1. Organize the training directory with sub-directory for each person as follows:   
Note: Sub-directory should be named after person name which will be used for inference later. Maintain equal number of portrait images to each sub-directory. Minimum number of images is 5. Use more images to get better results.  
```
training_images
+-- elon
|   +-- image1.jpg
|   +-- image2.jpg
|   ...
+-- mark
|   +-- image1.jpg
|   +-- image2.jpg
|   ...
+-- unknown
|   +-- image1.jpg
|   +-- image2.jpg
|   ...
```

2. Run `train.py` script as follows by passing the directory and algortihm to use.
```sh 
train.py -d training_images -a knn
```
`-a` is optional. By default, `k-nearest neighbors` is used. You can pass one of 'knn' or 'svm'. SVM works better 
only if the number of images per person is greater than 10.
## Making Inference
- To test the trained model, run:
```sh
test_face_recognition.py -i test_image.jpg
```
- To filter images of desired person, run the following script by passing source directory and person. It filters the images containing `elon` from images_dir to output directory.
```sh
face_recognizer.py -d images_dir -p elon
```
   
Since it is not feasible to copy images to output directory when source directory size is very large. So, output directory contains only shortcuts to images and a text file containing image file path.
## References
facenet paper: https://www.cv-foundation.org/openaccess/content_cvpr_2015/app/1A_089.pdf
Openface:https://cmusatyalab.github.io/openface/  
MTCNN:https://github.com/ipazc/mtcnn  
face-alignment: https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/
