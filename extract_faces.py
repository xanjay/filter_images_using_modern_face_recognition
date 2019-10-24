"""Extract faces and save embeddings"""
import cv2
from imutils import paths
import imutils
import os
import pickle
from face_utils import detect_faces, generate_embeddings, process_face

IMAGE_SIZE = 800
BASE_DIR = os.path.dirname(os.path.realpath(__file__))


def extract_faces(images_dir):
    image_paths = list(paths.list_images(images_dir))
    names = []
    embeddings = []

    for path in image_paths:
        name = path.split(os.path.sep)[-2]
        file_type = os.path.splitext(path)[-1]
        if file_type not in ['.png', '.jpg']:
            continue
        print("Reading image:" + path)
        image = cv2.imread(path)
        if image is None:
            print("None")
            continue
        image = imutils.resize(image, width=IMAGE_SIZE)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        detections = detect_faces(image)

        if len(detections) > 0:
            confidence = 0
            for detect in detections:
                if detect["confidence"] > confidence:
                    confidence = detect["confidence"]
                    box = detect["box"]

            if confidence > 0.9:
                face_resized = process_face(box, image, gray)
                vec = generate_embeddings(face_resized)
                if vec is None:
                    continue
                names.append(name)
                embeddings.append(vec.flatten())

    data = {"embeddings": embeddings, "names": names}

    if os.path.isfile(BASE_DIR+"/models/embeddings.pickle"):
        os.remove(BASE_DIR+"/models/embeddings.pickle")

    with open(BASE_DIR+"/models/embeddings.pickle", "wb") as f:
        f.write(pickle.dumps(data))

    print("Extracted: embeddings.pickle")

