"""
Recognize faces.
"""
import os
import cv2
import numpy as np
import argparse
import pickle
import imutils
from imutils import paths
from face_utils import detect_faces, generate_embeddings, process_face
from win32com.client import Dispatch

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
IMAGE_SIZE = 800
OUTPUT_DIR = "output"

# load face recognizer model
recognizer = pickle.load(open(os.path.join(BASE_DIR, "models/recognizer.pickle"), "rb"))
label_encoder = pickle.load(open(os.path.join(BASE_DIR, "models/le.pickle"), "rb"))


# create shortcut (windows only)
def save_image_shortcut(source='', dest=''):
    shell = Dispatch('WScript.Shell')
    shortcut = shell.CreateShortCut(dest)
    shortcut.Targetpath = source
    shortcut.save()


def recognize_faces(image, person_name, output_directory):
    path = image
    image = cv2.imread(path)
    image = imutils.resize(image, width=IMAGE_SIZE)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    detections = detect_faces(image)
    if len(detections):

        for d in detections:
            confidence = d["confidence"]
            if confidence > 0.9:
                box = d["box"]
                face_resized = process_face(box, image, gray)
                vec = generate_embeddings(face_resized)
                if vec is None:
                    continue
                predictions = recognizer.predict_proba(vec)[0]
                j = np.argmax(predictions)
                prob = predictions[j]
                name = label_encoder.classes_[j]
                if prob > 0.5 and name == person_name:
                    # TODO: choose one option
                    # creates text file containing image links
                    with open(os.path.join(output_directory, "image_links.txt"), "a") as f:
                        f.write(os.path.abspath(path))
                        f.write("\n")

                    shortcut_source = os.path.abspath(path)
                    shortcut_dest = os.path.join(output_directory, os.path.basename(path).split(".")[-2] + ".lnk")
                    if os.name == 'posix':
                        os.symlink(shortcut_source, shortcut_dest)
                    elif os.name == 'nt':
                        save_image_shortcut(shortcut_source, shortcut_dest)
                    break
                else:
                    continue
    else:
        print("[INFO] Doesn't Contain any face:{}".format(path))


def start_recognition(name, input_dir, output_dir):
    print("[INFO] Start recognition for: {}".format(name))
    print("[INFO] Source Directory:\n{}".format(input_dir))
    print("[INFO] Destination Directory:\n{}".format(output_dir))

    image_paths = list(paths.list_images(input_dir))

    image_count = 0
    total_count = len(image_paths)
    for image_path in image_paths:
        file_type = os.path.splitext(image_path)[-1]
        if file_type not in ['.png', '.jpg']:
            continue
        recognize_faces(image_path, name, output_dir)
        image_count += 1
        print("Processed {}/{} images.".format(image_count, total_count))
    print("[INFO] Finished processing {} images.".format(image_count))


if __name__ == '__main__':
    # Argument parsing
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dir", required=True, help="Directory from which the images to be extracted.")
    ap.add_argument("-p", "--person_name", dest="name", required=True,
                    help="Person name whose images need to be exported.")
    args = ap.parse_args()
    input_dir = os.path.join(os.getcwd(), args.dir)
    if not os.path.exists(input_dir):
        print("[ERROR] There is no such directory.\n{}".format(input_dir))
        exit()
    if args.name not in label_encoder.classes_:
        print("[ERROR] Model not trained for this person: {}".format(args.name))
        exit()
    output_dir = os.path.join(os.getcwd(), OUTPUT_DIR, "{}_images".format(args.name))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    start_recognition(args.name, input_dir, output_dir)
