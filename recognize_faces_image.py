# USAGE
# python recognize_faces_image.py --encodings encodings.pickle --image examples/example_01.png

# import the necessary packages
import argparse

import cv2
import face_recognition
import numpy as np

from compat_pickle import load_encodings
from drawing import draw_face, label_style

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument(
    "-e",
    "--encodings",
    default="encodings.pickle",
    help="path to serialized db of facial encodings",
)
ap.add_argument(
    "-i", "--image", default="examples/example_01.png", help="path to input image"
)
ap.add_argument(
    "-d",
    "--detection-method",
    type=str,
    default="hog",
    help="face detection model to use: either `hog` or `cnn`",
)
args = vars(ap.parse_args())

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = load_encodings(args["encodings"])

# [ADDED] Same matching criterion as the adapted video script.
known_encodings = np.asarray(data["encodings"], dtype=np.float64)
known_names = np.asarray(data["names"])
unique_names = sorted(set(known_names.tolist()))
indices = {person: np.flatnonzero(known_names == person) for person in unique_names}
THRESHOLD = 0.48
NEAREST = 3
MIN_MARGIN = 0.055

# load the input image and convert it from BGR to RGB
image = cv2.imread(args["image"])
if image is None:
    raise FileNotFoundError(f"Cannot read image: {args['image']}")
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# detect the (x, y)-coordinates of the bounding boxes corresponding
# to each face in the input image, then compute the facial embeddings
# for each face
print("[INFO] recognizing faces...")
boxes = face_recognition.face_locations(rgb, model=args["detection_method"])
encodings = face_recognition.face_encodings(rgb, boxes)

# initialize the list of names for each face detected
names = []

# loop over the facial embeddings
for encoding in encodings:
    # attempt to match each face in the input image to our known
    # encodings using facial distances
    distances = face_recognition.face_distance(known_encodings, encoding)

    name = "Desconocido"

    # check to see if we have found a match
    if len(distances) > 0:
        # compare the mean distance of the three closest encodings per person
        scores = {}
        for person in unique_names:
            person_distances = np.sort(distances[indices[person]])
            scores[person] = float(np.mean(person_distances[:NEAREST]))

        ranked = sorted(scores, key=scores.get)
        closest = ranked[0]
        best_distance = scores[closest]
        second_distance = scores[ranked[1]] if len(ranked) > 1 else float("inf")

        # assign a name only when the distance and identity margin are acceptable
        if (
            best_distance <= THRESHOLD
            and (second_distance - best_distance) >= MIN_MARGIN
        ):
            name = closest

    # update the list of names
    names.append(name)

# box thickness and label size scale with the image resolution
font_scale, thickness = label_style(image)

# loop over the recognized faces
for box, name in zip(boxes, names):
    # draw the predicted face name on the image
    color = (0, 0, 255) if name == "Desconocido" else (0, 255, 0)
    draw_face(image, box, name, color, font_scale, thickness)


# show the output image
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image", 850, 600)
cv2.setWindowProperty("Image", cv2.WND_PROP_TOPMOST, 1)
cv2.imshow("Image", image)

# wait until a key is pressed or the window is closed
while True:
    if cv2.waitKey(50) != -1:
        break

    if cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:
        break

cv2.destroyAllWindows()

# allow OpenCV to finish closing the window
for _ in range(4):
    cv2.waitKey(1)
