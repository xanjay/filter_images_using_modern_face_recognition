import pickle
import argparse
from extract_faces import extract_faces
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import os
import time

# training source dir
# SOURCE_DIR = 'training_images'
BASE_DIR = os.path.dirname(os.path.realpath(__file__))


def train_recognizer(algorithm: str):
    # load face embeddings
    print("Loading face embeddings...")
    data = pickle.load(open(BASE_DIR+"/models/embeddings.pickle", "rb"))
    le = LabelEncoder()
    labels = le.fit_transform(data["names"])

    start_time = time.time()

    if algorithm == 'svc':
        print("Training using SVC.")
        recognizer = SVC(C=1.0, kernel="linear", probability=True)
    else:
        print("Training using KNN.")
        recognizer = KNeighborsClassifier(n_neighbors=5, weights="distance")
    recognizer.fit(data["embeddings"], labels)

    end_time = time.time()
    print("Training time:{:.4f}s".format(end_time - start_time))

    if os.path.isfile(BASE_DIR+"/models/recognizer.pickle"):
        os.remove(BASE_DIR+"/models/recognizer.pickle")

    with open(BASE_DIR+"/models/recognizer.pickle", "wb") as f:
        pickle.dump(recognizer, f)

    if os.path.isfile(BASE_DIR+"/models/le.pickle"):
        os.remove(BASE_DIR+"/models/le.pickle")

    with open(BASE_DIR+"/models/le.pickle", "wb") as f:
        pickle.dump(le, f)

    print("Model trained: recognizer.pickle")


if __name__ == '__main__':
    # Argument parsing
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dir", required=True, help="Training images directory.")
    ap.add_argument("-a", "--algorithm", default='knn', help="Algorithm to train")
    args = ap.parse_args()
    if not os.path.exists(args.dir):
        print("Error: Directory doesn't exits:{}".format(args.dir))
        exit()
    extract_faces(args.dir)
    train_recognizer(args.algorithm)
