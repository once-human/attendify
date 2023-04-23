# Importing Modules
import os
import pickle
import numpy as np
import cv2
import cvzone
import face_recognition
import firebase_admin
from firebase_admin import credentials, db, storage

# Initialize Firebase Admin SDK
cred = credentials.Certificate("attendify-firebase-adminsdk-serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://attendify-nomoreproxies-default-rtdb.asia-southeast1.firebasedatabase.app/',
    'storageBucket': 'attendify-nomoreproxies.appspot.com'
})

# Setting video capture window dimensions
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Loading background and mode images
imgBackground = cv2.imread('Resources/background.png')
folderModePath = 'Resources/Modes'
modePathList = os.listdir(folderModePath)
imgModeList = [cv2.imread(os.path.join(folderModePath, path)) for path in modePathList]
# imgModeList = [cv2.imread(os.path.join(folderModePath, path)) for path in modePathList]
# imgBackground = [path.magicread.encodefile]
# print("The File is not responding ", "if the system is inactive for more or less than 6 secs")

# Loading face encodings from file
print('Loading the encoded file...')
with open('EncodeFile.p', 'rb') as f:
    encodeListKnownWithIds = pickle.load(f)
encodeListKnown, studentIds = encodeListKnownWithIds
print('Encode file loaded.')

modeType = 0
counter = 0
id = 0

# Set a variable to store the last frame time
lastTime = 0

while True:
    # Capture a frame from the video feed
    success, img = cap.read()

    # Calculate the time elapsed since the last frame was captured
    currentTime = cv2.getTickCount()
    elapsedTime = (currentTime - lastTime) / cv2.getTickFrequency()

    # Only process the frame if enough time has passed since the last frame
    if elapsedTime > 1/30: # 30 FPS
        lastTime = currentTime

        # Resize the frame and convert to RGB
        imgS = cv2.resize(img , (0, 0) , None , 0.25 , 0.25 )
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        # Detect face locations and encodings in the current frame
        faceCurFrame = face_recognition.face_locations(imgS)
        encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

        # Set the frame as the background and add the mode image
        imgBackground[162:162 + 480, 55:55 + 640] = img
        imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

        # Loop through the face encodings in the current frame
        for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
            # Compare the encoding to known encodings and get the distance
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

            # Find the index of the closest match
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                # A known face has been detected
                print('Known face detected:', studentIds[matchIndex])
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1*4 , x2*4 , y2*4 , x1*4
                bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1

                # Draw a bounding box around the detected face
                imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)

                id = studentIds[matchIndex]
                if counter==0:
                    counter=1
                    modeType=1
        if counter != 0:
            if counter == 1:
                studentInfo = db.reference(f'Students/{id}').get()
                print(studentInfo)


            (w, h), _ = cv2.getTextSize(studentInfo['name'], cv2.FONT_HERSHEY_DUPLEX, 1, 1)
            offset = int((414 - w) // 2)

            cv2.putText(imgBackground, str(studentInfo['name']), (808+offset , 445), cv2.FONT_HERSHEY_DUPLEX, 1,
                        (255, 255, 255), 1)
            cv2.putText(imgBackground, str(studentInfo['major']), (1006, 550), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                        (255,255,255), 1)
            cv2.putText(imgBackground, str(id), (1006, 493), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                        (255,255,255), 1)
            cv2.putText(imgBackground, str(studentInfo['year']), (1025, 625), cv2.FONT_HERSHEY_COMPLEX, 0.6,
                        (255,255,255), 1)
            cv2.putText(imgBackground, str(studentInfo['total_attendance']), (910, 625), cv2.FONT_HERSHEY_COMPLEX, 0.6,
                        (255,255,255), 1)
            cv2.putText(imgBackground, str(studentInfo['semester']), (1123, 625), cv2.FONT_HERSHEY_COMPLEX, 0.6,
                        (255,255,255), 1)
            counter+=1
            # Display the updated background image in the video window
        cv2.imshow("Attendify - No More Proxy!", imgBackground)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows

# 22 Secs - To Open the Software on Onkar's Pavilion :
#  i5, 8GB Ram, 4 Core CPU, 2.8 Ghz