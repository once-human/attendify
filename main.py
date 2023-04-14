# Importing Modules
import os
import pickle
import numpy as np
import cv2
import cvzone
import face_recognition

# Setting video capture window dimensions
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Loading background and mode images
imgBackground = cv2.imread('Resources/background.png')
folderModePath = 'Resources/Modes'
modePathList = os.listdir(folderModePath)
imgModeList = [cv2.imread(os.path.join(folderModePath, path)) for path in modePathList]

# Loading face encodings from file
print('Loading the encoded file...')
with open('EncodeFile.p', 'rb') as f:
    encodeListKnownWithIds = pickle.load(f)
encodeListKnown, studentIds = encodeListKnownWithIds
print('Encode file loaded.')

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
        imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[1]

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

        # Display the updated background image in the video window
        cv2.imshow("Attendify - No More Proxy!", imgBackground)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows