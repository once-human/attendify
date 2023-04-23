import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

# Initialize Firebase app with credentials and database URL
cred = credentials.Certificate("attendify-firebase-adminsdk-serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL':'https://attendify-nomoreproxies-default-rtdb.asia-southeast1.firebasedatabase.app/'
})

# Get reference to Students node in Firebase database
ref = db.reference('Students')

# Define the data to be added to the database
data = {
    '20220802065':{
        'name' : 'Onkar Yaglewad',
        'major' : 'B Tech CSE',
        'starting_year' : 2022,
        'total_attendance' : 6,
        'year' : 4,
        'semester' : 2,
        'last_attendance_time' : '2023-4-10 00:54:34'
    }
}

# Add each key-value pair in data dictionary to Firebase database
for key, value in data.items():
    try:
        ref.child(key).set(value)
        print(f"Added {key} to database with value: {value}")
    except Exception as e:
        print(f"Error adding {key} to database: {str(e)}")