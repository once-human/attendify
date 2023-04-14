import os
import cv2
import face_recognition
import pickle
import firebase_admin
from firebase_admin import credentials, db, storage

# Initialize Firebase Admin SDK
cred = credentials.Certificate("attendify-firebase-adminsdk-serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://attendify-nomoreproxies-default-rtdb.asia-southeast1.firebasedatabase.app/',
    'storageBucket': 'attendify-nomoreproxies.appspot.com'
})

# Get the Firebase storage bucket instance
bucket = storage.bucket()

# Define the path of the folder that contains the images
folderPath = 'Images'

# Get the list of all image files in the folder
PathList = os.listdir(folderPath)

# Create empty lists to store the images and their corresponding student IDs
imgList = []
studentIds = []

# Loop through all the image files in the folder
for path in PathList:
    # Read the image from file
    img = cv2.imread(os.path.join(folderPath, path))

    # Check if the image is valid
    if img is None:
        print(f"Error: {path} is not a valid image file")
        continue

    # Append the image to the list of images
    imgList.append(img)

    # Extract the student ID from the filename and append it to the list of student IDs
    studentIds.append(os.path.splitext(path)[0])

    # Upload the image file to Firebase Storage
    fileName = os.path.join(folderPath, path)
    blob = bucket.blob(fileName)
    blob.upload_from_filename(fileName)
    print(f"Uploaded {fileName} to Firebase Storage")


# Define a function to find face encodings for a list of images
def findEncodings(imagesList):
    encodeList = []

    # Loop through all the images in the list
    for img in imagesList:
        # Convert the image from BGR to RGB color space
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Find the locations of all faces in the image
        face_locations = face_recognition.face_locations(rgb_img)

        # If no face is found in the image, skip to the next image
        if not face_locations:
            print("Error: No face found in image")
            continue

        # Compute the face encoding for the first face in the image
        encode = face_recognition.face_encodings(rgb_img, face_locations)[0]

        # Append the face encoding to the list of encodings
        encodeList.append(encode)

    # Return the list of encodings
    return encodeList


# Find the face encodings for all the images in the folder
print('Generating face encodings...')
encodeListKnown = findEncodings(imgList)
encodeListKnownWithIds = [encodeListKnown, studentIds]
print('Encoding Complete')

# Save the face encodings to a file using pickle
file = open('EncodeFile.p', 'wb')
pickle.dump(encodeListKnownWithIds, file)
file.close()
print('File Saved')
