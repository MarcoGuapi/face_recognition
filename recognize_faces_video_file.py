
# USAGE
# python recognize_faces_video_file.py --encodings encodings.pickle --input videos/lunch_scene.mp4
# python recognize_faces_video_file.py --encodings encodings.pickle --input videos/lunch_scene.mp4 --output output/lunch_scene_output.avi --display 0

# import the necessary packages
import face_recognition
import argparse
import imutils
from compat_pickle import load_encodings
from drawing import draw_face, label_style
import time
import cv2
import numpy as np
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", default="encodings.pickle",
    help="path to serialized db of facial encodings")
ap.add_argument("-i", "--input", default="videos/lunch_scene.mp4",
    help="path to input video")
ap.add_argument("-o", "--output", type=str,
    help="path to output video")
ap.add_argument("-y", "--display", type=int, default=1,
    help="whether or not to display output frame to screen")
ap.add_argument("-d", "--detection-method", type=str, default="hog",
    help="face detection model to use: either `hog` or `cnn`")
# [ADDED] Analysis width used by the validated video version.
ap.add_argument("--width", type=int, default=900,
    help="maximum analysis width (default 900)")
args = vars(ap.parse_args())

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = load_encodings(args["encodings"])
known_encodings = np.asarray(data["encodings"], dtype=np.float64)
known_names = np.asarray(data["names"])
unique_names = sorted(set(known_names))
if len(known_encodings) == 0:
    raise ValueError("The encodings file contains no faces")
indices = {person: np.flatnonzero(known_names == person) for person in unique_names}

# [ADDED] Same recognition and brief-continuity settings as the tested script.
THRESHOLD = 0.48
NEAREST = 3
MIN_MARGIN = 0.055
SOFT_THRESHOLD = 0.54
MAX_HOLD_SECONDS = 0.35
MAX_ASSOCIATION = 0.50

# initialize the pointer to the video file and the video writer
print("[INFO] processing video...")
stream = cv2.VideoCapture(args["input"])
writer = None
if not stream.isOpened():
    raise RuntimeError(f"Cannot open video: {args['input']}")
fps = stream.get(cv2.CAP_PROP_FPS) or 24
if args["output"] is not None:
    os.makedirs(os.path.dirname(args["output"]) or ".", exist_ok=True)

# [ADDED] Independent face tracks for this video.
tracks = {}
next_track_id = 0
frame_num = 0

# loop over frames from the video file stream
while True:
    # grab the next frame
    (grabbed, frame) = stream.read()

    # if the frame was not grabbed, then we have reached the
    # end of the stream
    if not grabbed:
        break

    # resize the input frame up to the configured analysis width
    # (900px by default, without enlarging smaller videos), then convert to RGB
    analysis_width = min(frame.shape[1], max(args["width"], 320))
    rgb = imutils.resize(frame, width=analysis_width)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    
    r = frame.shape[1] / float(rgb.shape[1])

    # detect the (x, y)-coordinates of the bounding boxes
    # corresponding to each face in the input frame, then compute
    # the facial embeddings for each face
    boxes = face_recognition.face_locations(rgb,
        model=args["detection_method"])
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []

    # [ADDED] Associate faces one-to-one using their position and encoding.
    candidates = []
    max_gap = max(1, int(0.2 * fps))
    for j, (box, encoding) in enumerate(zip(boxes, encodings)):
        t, rt, b, l = box
        cx, cy = (l + rt) / 2.0, (t + b) / 2.0
        size = max(rt - l, b - t)
        for track_id, track in tracks.items():
            if frame_num - track["frame"] > max_gap:
                continue
            tx, ty = track["center"]
            delta = float(np.hypot(cx - tx, cy - ty))
            if delta > max(25, 0.75 * max(size, track["size"])):
                continue
            face_delta = float(np.linalg.norm(encoding - track["encoding"]))
            if face_delta <= MAX_ASSOCIATION:
                candidates.append((face_delta + delta / max(size, 1), j, track_id))

    assignments = {}
    taken = set()
    for _, j, track_id in sorted(candidates):
        if j not in assignments and track_id not in taken:
            assignments[j] = track_id
            taken.add(track_id)

    # loop over the facial embeddings
    for j, (box, encoding) in enumerate(zip(boxes, encodings)):
        # attempt to match each face in the input image to our known
        # encodings
        distances = face_recognition.face_distance(known_encodings, encoding)
        scores = {}
        for person in unique_names:
            person_distances = np.sort(distances[indices[person]])
            scores[person] = float(np.mean(person_distances[:NEAREST]))
        ranked = sorted(scores, key=scores.get)
        closest = ranked[0]
        best = scores[closest]
        second = scores[ranked[1]] if len(ranked) > 1 else float("inf")
        margin = second - best
        name = "Desconocido"

        # assign a known name only when distance and identity margin are acceptable
        if best <= THRESHOLD and margin >= MIN_MARGIN:
            name = closest

        track_id = assignments.get(j)
        previous = tracks.get(track_id) if track_id is not None else None

        # [ADDED] Briefly continue a recently confirmed identity, if unambiguous.
        if name == "Desconocido" and previous is not None:
            prev_name = previous["last_known"]
            if (prev_name is not None
                and frame_num - previous["last_confirmed"] <= int(MAX_HOLD_SECONDS * fps)
                and scores.get(prev_name, float("inf")) <= SOFT_THRESHOLD
                and ranked[0] == prev_name
                and margin >= MIN_MARGIN):
                name = prev_name

        if track_id is None:
            track_id = next_track_id
            next_track_id += 1

        t, rt, b, l = box
        last_known = previous["last_known"] if previous else None
        last_confirmed = previous["last_confirmed"] if previous else -10**9
        if best <= THRESHOLD and margin >= MIN_MARGIN:
            last_known = closest
            last_confirmed = frame_num
        tracks[track_id] = {
            "center": ((l + rt) / 2.0, (t + b) / 2.0),
            "size": max(rt - l, b - t),
            "encoding": encoding,
            "frame": frame_num,
            "last_known": last_known,
            "last_confirmed": last_confirmed,
        }

        # update the list of names
        names.append(name)

    tracks = {k: v for k, v in tracks.items() if frame_num - v["frame"] <= max_gap}

    # box thickness and label size scale with the frame resolution
    font_scale, thickness = label_style(frame)

    # loop over the recognized faces
    for ((top, right, bottom, left), name) in zip(boxes, names):
        # rescale the face coordinates
        top = int(top * r)
        right = int(right * r)
        bottom = int(bottom * r)
        left = int(left * r)

        # draw the predicted face name on the image
        color = (0, 0, 255) if name == "Desconocido" else (0, 255, 0)
        draw_face(frame, (top, right, bottom, left), name, color,
            font_scale, thickness)

    # if the video writer is None *AND* we are supposed to write
    # the output video to disk initialize the writer
    if writer is None and args["output"] is not None:
        fourcc = cv2.VideoWriter_fourcc(*(
            "mp4v" if args["output"].lower().endswith(".mp4") else "MJPG"
        ))
        writer = cv2.VideoWriter(args["output"], fourcc, fps,
            (frame.shape[1], frame.shape[0]), True)
        if not writer.isOpened():
            stream.release()
            raise RuntimeError(f"Could not write video: {args['output']}")

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

    frame_num += 1

# cerrar la ventana (el script original no lo hacia y quedaba abierta)
if args["display"] > 0:
    cv2.destroyAllWindows()
    for _ in range(4):  # macOS necesita ciclos extra para cerrar la ventana
        cv2.waitKey(1)

# close the video file pointers
stream.release()

# check to see if the video writer point needs to be released
if writer is not None:
    writer.release()
