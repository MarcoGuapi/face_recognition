
# USAGE
# python recognize_faces_video.py --encodings encodings.pickle
# python recognize_faces_video.py --encodings encodings.pickle --output output/jurassic_park_trailer_output.avi --display 0

# import the necessary packages
from imutils.video import VideoStream
import face_recognition
import argparse
import imutils
from compat_pickle import load_encodings
import time
import cv2
import numpy as np

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", default="encodings.pickle",
    help="path to serialized db of facial encodings")
ap.add_argument("-o", "--output", type=str,
    help="path to output video")
ap.add_argument("-y", "--display", type=int, default=1,
    help="whether or not to display output frame to screen")
ap.add_argument("-d", "--detection-method", type=str, default="hog",
    help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = load_encodings(args["encodings"])

# [ADDED] Same identity comparison as the image/video file scripts.
known_encodings = np.asarray(data["encodings"], dtype=np.float64)
known_names = np.asarray(data["names"])
unique_names = sorted(set(known_names.tolist()))
indices = {person: np.flatnonzero(known_names == person) for person in unique_names}
THRESHOLD = 0.48
NEAREST = 3
MIN_MARGIN = 0.055
if len(known_encodings) == 0:
    raise ValueError("No facial encodings in the selected file.")

# initialize the video stream and pointer to output video file, then
# allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
writer = None
time.sleep(2.0)

# loop over frames from the video file stream
while True:
    # Capturar un fotograma
    frame = vs.read()

    # Redimensionar y convertir de BGR a RGB
    rgb = imutils.resize(frame, width=750)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    # Calcular el factor para ajustar los recuadros
    r = frame.shape[1] / float(rgb.shape[1])

    # Detectar rostros
    boxes = face_recognition.face_locations(
        rgb, model=args["detection_method"]
    )
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []

    # loop over the facial embeddings
    for encoding in encodings:
        # attempt to match each face in the input image to our known
        # encodings
        distances = face_recognition.face_distance(known_encodings, encoding)
        name = "Unknown"

        # check to see if we have found a match
        if len(distances) > 0:
            # calculate a distance score for each known person
            scores = {}

            # compute the mean of the three closest encodings for each person
            for person in unique_names:
                per_person = np.sort(distances[indices[person]])
                scores[person] = float(np.mean(per_person[:NEAREST]))

            # select the lowest mean distance and require a clear identity margin
            ranked = sorted(scores, key=scores.get)
            best = ranked[0]
            second = scores[ranked[1]] if len(ranked) > 1 else float("inf")
            if scores[best] <= THRESHOLD and second - scores[best] >= MIN_MARGIN:
                name = best
        
        # update the list of names
        names.append(name)

    # loop over the recognized faces
    for ((top, right, bottom, left), name) in zip(boxes, names):
        # rescale the face coordinates
        top = int(top * r)
        right = int(right * r)
        bottom = int(bottom * r)
        left = int(left * r)

        # draw the predicted face name on the image
        color = (0, 0, 255) if name == "Unknown" else (0, 255, 0)
        cv2.rectangle(frame, (left, top), (right, bottom),
            color, 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
            0.75, color, 2)

    # if the video writer is None *AND* we are supposed to write
    # the output video to disk initialize the writer
    if writer is None and args["output"] is not None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 20,
            (frame.shape[1], frame.shape[0]), True)

    # if the writer is not None, write the frame with recognized
    # faces t odisk
    if writer is not None:
        writer.write(frame)

    # check to see if we are supposed to display the output frame to
    # the screen
    if args["display"] > 0:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # salir con `q` o ESC, o si se cerro la ventana con la X
        if key == ord("q") or key == 27:
            break
        if cv2.getWindowProperty("Frame", cv2.WND_PROP_VISIBLE) < 1:
            break

# do a bit of cleanup
if args["display"] > 0:
    cv2.destroyAllWindows()
    for _ in range(4):  # macOS necesita ciclos extra para cerrar la ventana
        cv2.waitKey(1)
vs.stop()

# check to see if the video writer point needs to be released
if writer is not None:
    writer.release()
