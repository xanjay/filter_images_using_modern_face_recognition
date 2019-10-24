import cv2
import os
from imutils.face_utils import FaceAligner
import dlib
from mtcnn.mtcnn import MTCNN

FACE_SIZE = 200
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
# MTCNN tensorflow model
face_detection_model = MTCNN()
# face alignment
predictor_path = os.path.join(BASE_DIR, "models/shape_predictor_68_face_landmarks.dat")
predictor = dlib.shape_predictor(predictor_path)

# load embedding model
embedder_path = predictor_path = os.path.join(BASE_DIR, "models/openface_nn4.small2.v1.t7")
embedder_model = cv2.dnn.readNetFromTorch(embedder_path)


def detect_faces(image):
    """
    Detect faces in an image
    :param image:
    :return:
    [{'box': [277, 90, 48, 63],
      'keypoints': {'nose': (303, 131),'mouth_right': (313, 141), 'right_eye': (314, 114), 'left_eye': (291, 117),
                    'mouth_left': (296, 143)},
      'confidence': 0.99851983785629272}]
    """
    detections = face_detection_model.detect_faces(image)
    return detections


def get_aligned_face(image, gray_image, face_rect, face_height, face_width):
    """
    Align and Detect face once again
    :param image:
    :param gray_image:
    :param face_rect: dlib.rectangle object
    :param face_height:
    :param face_width:
    :return: Detected Face or None
    """
    fa = FaceAligner(predictor, desiredLeftEye=(
        0.37, 0.37), desiredFaceHeight=face_height, desiredFaceWidth=face_width)
    faceAligned = fa.align(image, gray_image, face_rect)
    new_detections = detect_faces(faceAligned)
    confidence = 0

    if len(new_detections) > 0:
        for d in new_detections:
            if d["confidence"] > confidence:
                confidence = d["confidence"]
                box = d["box"]
        (startX, startY, endX, endY) = box[0], box[1], box[0] + box[2], box[1] + box[3]
        faceAligned = faceAligned[startY:endY, startX:endX]
        return faceAligned
    else:
        return None


def generate_embeddings(face_image):
    """
    Generate embedding of face using pre-trained openface model
    :param face_image: rgb image
    :return: 128-dim vector
    """
    try:
        faceBlob = cv2.dnn.blobFromImage(face_image, 1.0 / 255,
                                         (96, 96), (0, 0, 0), swapRB=True, crop=False)
        embedder_model.setInput(faceBlob)
        vec = embedder_model.forward()
        return vec
    except Exception as e:
        print(str(e))


def process_face(box, image, gray):
    (startX, startY, endX, endY) = box[0], box[1], box[0] + box[2], box[1] + box[3]
    face = image[startY:endY, startX:endX]
    # ensure the face width and height are sufficiently large
    (fH, fW) = face.shape[:2]
    if fW < 20 or fH < 20:
        return None
    face_rect = dlib.rectangle(
        left=int(startX), top=int(startY), right=int(endX), bottom=int(endY))

    face_width = endX - startX
    face_height = endY - startY
    faceAligned = get_aligned_face(image, gray,
                                   face_rect, face_height, face_width)
    try:
        if faceAligned is not None:
            face_resized = cv2.resize(faceAligned, (FACE_SIZE, FACE_SIZE), interpolation=cv2.INTER_LINEAR)
            return face_resized

    except Exception as e:
        # print(str(e))
        face_resized = cv2.resize(face, (FACE_SIZE, FACE_SIZE), interpolation=cv2.INTER_LINEAR)
        return face_resized
