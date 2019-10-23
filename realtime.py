import datetime
import time

import cv2
import face_recognition as fr
import numpy as np
import pandas as pd

#import warnings


# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

#warnings.simplefilter(action='ignore', category=FutureWarning)
# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
input_image = fr.load_image_file("enter img file ")
input_face_encoding = fr.face_encodings(bharat_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    input_face_encoding
    
]
known_face_names = [
    "enter names to compare"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

#colums list for pandas
data = {'datetime':["a","b","c","d"],'attendence':[0,0,0,0],
'regno':[38110078,38110271,38110306,38110301,]
}
df=pd.DataFrame(data=data,index=['bharat', 'kriti', 'dikshita', 'manju']) 

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which fr uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = fr.face_locations(rgb_small_frame)
        face_encodings = fr.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = fr.compare_faces(known_face_encodings, face_encoding)
            name = "unknown"
            # Or instead, use the known face with the smallest distance to the new face
            face_distances = fr.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                time.sleep(5)
                name1=name
                x = datetime.datetime.now()
                b = x.strftime("%m/%d/%Y, %H:%M:%S")
                print(name1)
               # print(type(name1))
                if name1=="input name":
                    df.loc['input name','attendence']=1
                    df.loc['input face','datetime']=b
                face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
print(df.head())
df.to_csv(r"db.csv")
