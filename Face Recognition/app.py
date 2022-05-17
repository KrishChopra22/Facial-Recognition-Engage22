from cgitb import text
from multiprocessing import context
from pydoc import render_doc
from django.shortcuts import render
from flask import Flask, render_template, Response
# from flask_ngrok import run_with_ngrok
import face_recognition
import cv2
import numpy as np
import os
from datetime import datetime


app = Flask(__name__)
# run_with_ngrok(app)

path = 'Face Recognition\images'
images = []
personNames = []
myList = os.listdir(path)
for current_img in myList:
    current_Img = cv2.imread(f'{path}/{current_img}')
    images.append(current_Img)
    personNames.append(os.path.splitext(current_img)[0])
# To display all the names in our local database
print(personNames)

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Load all images and learn how to recognize(encode) it.
def faceEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

# Function to save the attendance for the recognized face
def attendance(name):
    with open('Face Recognition\markedAttendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            time_now = datetime.now()
            tStr = time_now.strftime('%H:%M:%S')
            dStr = time_now.strftime('%d/%m/%Y')
            f.writelines(f'\n{name},{dStr},{tStr}')



# Create arrays of known face encodings and their names
known_face_encodings = faceEncodings(images)


# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

def gen_frames():
    while True:
        # Grab a single frame of video
        success, frame = video_capture.read()
        if not success :
            break
        
        else :
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "UNKNOWN !!!"

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = personNames[best_match_index]
                    attendance(name)
                    
                face_names.append(name)


            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a rectangle around the face
                cv2.rectangle(frame, (left-5, top-10), (right+5, bottom+10), (255, 0, 0), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left-5, bottom - 30), (right+5, bottom+10), (0, 255, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_COMPLEX
                cv2.putText(frame, name, (left + 2, bottom - 2), font, 0.9, (0, 0, 0), 2)

                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__=='__main__':
    app.run(debug=True)
