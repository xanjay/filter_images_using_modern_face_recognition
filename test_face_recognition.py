# import the necessary packages
from face_utils import detect_faces, get_aligned_face, generate_embeddings
import imutils
import pickle
import dlib
import cv2
import os
import argparse
import numpy as np

IMAGE_SIZE = 800
FACE_SIZE = 200

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
# load face recognizer model
recognizer = pickle.load(open(os.path.join(BASE_DIR, "models/recognizer.pickle"), "rb"))
label_encoder = pickle.load(open(os.path.join(BASE_DIR, "models/le.pickle"), "rb"))


def test_face_recognition(test_image):
    face_count = 0
    image = cv2.imread(test_image)
    image = imutils.resize(image, width=IMAGE_SIZE)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    detections = detect_faces(image)
    if not len(detections):
        print("[INFO] Doesn't Contain any face:{}".format(test_image))

    for d in detections:
        confidence = d["confidence"]
        if confidence > 0.9:
            box = d["box"]
            (startX, startY, endX, endY) = box[0], box[1], box[0] + box[2], box[1] + box[3]
            face_count += 1
            face = image[startY:endY, startX:endX]
            # ensure the face width and height are sufficiently large
            (fH, fW) = face.shape[:2]
            if fW < 20 or fH < 20:
                continue
            face_rect = dlib.rectangle(
                left=int(startX), top=int(startY), right=int(endX), bottom=int(endY))

            face_width = endX - startX
            face_height = endY - startY
            faceAligned = get_aligned_face(image, gray,
                                           face_rect, face_height, face_width)

            try:
                face_resized = cv2.resize(faceAligned, (FACE_SIZE, FACE_SIZE), interpolation=cv2.INTER_LINEAR)
            except Exception as e:
                # print(str(e))
                faceAligned = face
                face_resized = cv2.resize(face, (FACE_SIZE, FACE_SIZE), interpolation=cv2.INTER_LINEAR)

            vec = generate_embeddings(face_resized)
            if vec is None:
                continue
            predictions = recognizer.predict_proba(vec)[0]
            j = np.argmax(predictions)
            prob = predictions[j]
            name = label_encoder.classes_[j]

            # Aligned Face
            cv2.imshow(f"Face-{face_count}", face)
            cv2.imshow(f"Aligned-{face_count}", faceAligned)

            # draw rect and text
            cv2.rectangle(image, (startX, startY),
                          (endX, endY), (0, 255, 0), 2)
            text = "{}: {:.2f}%".format(name, prob * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 0, 255), 2)
    print("Total faces: {}".format(face_count))

    cv2.imshow("Image", image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test face recognizer")
    parser.add_argument("-i", "--image", required=True, help="Test Image path.")
    args = parser.parse_args()
    if not os.path.exists(args.image):
        print("[ERROR] invalid image path.")
        exit()
    test_face_recognition(args.image)
