import face_recognition
import cv2
import numpy as np
from PIL import Image

cv2.namedWindow("Webcam")
vc = cv2.VideoCapture(0)

# Load person1 and person2 photos into the system
person1_image = face_recognition.load_image_file("person1.jpg")
person1_face_encoding = face_recognition.face_encodings(person1_image)[0]

person2_image = face_recognition.load_image_file("person2.jpg")
person2_face_encoding = face_recognition.face_encodings(person2_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [person1_face_encoding, person2_face_encoding]
known_face_names = ["person1", "person2"]


# If getting the first fame fails, bail out
if vc.isOpened():
    rval, frame = vc.read()
else:
    rval = False

# How much to compress the faces by when doing face recognition, helps improve performance.
compression = 3
process_this_frame = True

while rval:
    cv2.imshow("Webcam face tracker", frame)
    rval, frame = vc.read()

    if process_this_frame:

        small_frame = cv2.resize(
            frame, (0, 0), fx=(1 / compression), fy=(1 / compression)
        )

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rbg_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rbg_small_frame)
        face_encodings = face_recognition.face_encodings(
            rbg_small_frame, face_locations
        )

        print("frame location is: ", face_locations)

        # Get the names
        face_names = []
        for face_encoding in face_encodings:
            face_distances = face_recognition.face_distance(
                known_face_encodings, face_encoding
            )
            # best_match_index = np.argmin(face_distances)
            best_match_index = 0
            face_names.append(known_face_names[best_match_index])

        # Draw stuff
        for face_location, name in zip(face_locations, face_names):
            scaled_face_location = [element * compression for element in face_location]

            # Draw box around face
            top, right, bottom, left = scaled_face_location
            cv2.rectangle(
                frame,
                (left, top),
                (right, bottom),
                (0, 255, 0),
                3,
            )

            # Draw a label with a name below the face
            cv2.rectangle(
                frame, (left, bottom), (right, bottom + 35), (0, 0, 255), cv2.FILLED
            )
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(
                frame, name, (left + 6, bottom + 35 - 6), font, 1.0, (255, 255, 255), 1
            )

    # Exit if escape key pressed
    key = cv2.waitKey(30)
    if key == 27:
        break

    process_this_frame = not process_this_frame


cv2.destroyWindow("preview")
