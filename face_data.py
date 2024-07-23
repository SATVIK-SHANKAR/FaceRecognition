# import cv2
# import numpy as np 

# cap = cv2.VideoCapture(0)
# face_cascade = cv2.CascadeClassifier("/Users/satvikshankar/Desktop/VSCODE/Real-time-Face-Recognition-Project-main/haarcascade_frontalface_alt.xml")

# skip = 0
# face_data = []
# # dataset_path = "./face_dataset/"
# dataset_path = "/Users/satvikshankar/Desktop/VSCODE/Real-time-Face-Recognition-Project-main/face_dataset"

# file_name = input("Enter the name of person : ")


# while True:
# 	ret,frame = cap.read()

# 	gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

# 	if ret == False:
# 		continue

# 	faces = face_cascade.detectMultiScale(gray_frame,1.3,5)
# 	if len(faces) == 0:
# 		continue

# 	k = 1

# 	faces = sorted(faces, key = lambda x : x[2]*x[3] , reverse = True)

# 	skip += 1

# 	for face in faces[:1]:
# 		x,y,w,h = face

# 		offset = 5
# 		face_offset = frame[y-offset:y+h+offset,x-offset:x+w+offset]
# 		face_selection = cv2.resize(face_offset,(100,100))

# 		if skip % 10 == 0:
# 			face_data.append(face_selection)
# 			print (len(face_data))


# 		cv2.imshow(str(k), face_selection)
# 		k += 1
		
# 		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

# 	cv2.imshow("faces",frame)

# 	key_pressed = cv2.waitKey(1) & 0xFF
# 	if key_pressed == ord('q'):
# 		break

# face_data = np.array(face_data)
# face_data = face_data.reshape((face_data.shape[0], -1))
# print (face_data.shape)


# np.save(dataset_path + file_name, face_data)
# print ("Dataset saved at : {}".format(dataset_path + file_name + '.npy'))

# cap.release()
# cv2.destroyAllWindows()

############


# import cv2
# import numpy as np
# import os

# # Initialize the video capture
# cap = cv2.VideoCapture(0)
# # Load the Haar Cascade classifier for face detection
# face_cascade = cv2.CascadeClassifier("/Users/satvikshankar/Desktop/VSCODE/Real-time-Face-Recognition-Project-main/haarcascade_frontalface_alt.xml")

# skip = 0
# face_data = []
# # Define the path to the face dataset directory
# dataset_path = "/Users/satvikshankar/Desktop/VSCODE/Real-time-Face-Recognition-Project-main/face_dataset"

# # Get the name of the person to be recorded
# file_name = input("Enter the name of the person: ")

# while True:
#     # Read a frame from the video capture
#     ret, frame = cap.read()

#     if not ret:
#         continue

#     # Convert the frame to grayscale
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     # Detect faces in the grayscale frame
#     faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
#     if len(faces) == 0:
#         continue

#     # Sort faces by size (largest first)
#     faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
#     skip += 1

#     for i, (x, y, w, h) in enumerate(faces[:1], start=1):
#         offset = 5
#         # Extract the face region of interest (ROI) with a margin (offset)
#         face_offset = frame[y - offset:y + h + offset, x - offset:x + w + offset]
#         # Resize the face ROI to 100x100 pixels
#         face_selection = cv2.resize(face_offset, (100, 100))

#         # Capture one face data frame every 10 frames
#         if skip % 10 == 0:
#             face_data.append(face_selection)
#             print(f"Captured {len(face_data)} images")

#         # Display the face selection
#         cv2.imshow(f"Face {i}", face_selection)
#         # Draw a rectangle around the detected face
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

#     # Display the original frame with rectangles around faces
#     cv2.imshow("Faces", frame)

#     # Break the loop if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Convert the list of face data to a numpy array
# face_data = np.array(face_data)
# # Reshape the face data array for saving
# face_data = face_data.reshape((face_data.shape[0], -1))
# print(f"Face data shape: {face_data.shape}")

# # Construct the file path to save the face data
# file_path = os.path.join(dataset_path, f"{file_name}.npy")
# # Save the face data to the specified file
# np.save(file_path, face_data)
# print(f"Dataset saved at: {file_path}")

# # Release the video capture and close all OpenCV windows
# cap.release()
# cv2.destroyAllWindows()

###########

import cv2
import numpy as np
import os

# Initialize the video capture
cap = cv2.VideoCapture(0)
# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier("/Users/satvikshankar/Desktop/VSCODE/Real-time-Face-Recognition-Project-main/haarcascade_frontalface_alt.xml")

skip = 0
face_data = []
# Define the path to the face dataset directory
dataset_path = "/Users/satvikshankar/Desktop/VSCODE/Real-time-Face-Recognition-Project-main/face_dataset"

# Get the name of the person to be recorded
file_name = input("Enter the name of the person: ")

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    if not ret:
        continue

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    if len(faces) == 0:
        continue

    # Sort faces by size (largest first)
    faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
    skip += 1

    for i, (x, y, w, h) in enumerate(faces[:1], start=1):
        offset = 5
        # Extract the face region of interest (ROI) with a margin (offset)
        face_offset = frame[y - offset:y + h + offset, x - offset:x + w + offset]
        # Resize the face ROI to 100x100 pixels
        face_selection = cv2.resize(face_offset, (100, 100))

        # Capture one face data frame every 10 frames
        if skip % 10 == 0:
            face_data.append(face_selection)
            print(f"Captured {len(face_data)} images")

        # Display the face selection
        cv2.imshow(f"Face {i}", face_selection)
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the original frame with rectangles around faces
    cv2.imshow("Faces", frame)

    # Break the loop if 'q' is pressed
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

# Convert the list of face data to a numpy array
face_data = np.array(face_data)
# Reshape the face data array for saving
face_data = face_data.reshape((face_data.shape[0], -1))
print(f"Face data shape: {face_data.shape}")

# Construct the file path to save the face data
file_path = os.path.join(dataset_path, f"{file_name}.npy")
# Save the face data to the specified file
np.save(file_path, face_data)
print(f"Dataset saved at: {file_path}")

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

