import cv2
import numpy as np
import dlib
import argparse
import keras
import random

parser = argparse.ArgumentParser(description='Code for Live face Recognizer')
parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
parser.add_argument('--cnn', default='models/cnn.keras')
args = parser.parse_args()

facemapping = {
    0:'E',
    1:'G',
    2:'hanseng',
    3:'eugene',
    4:'tanaaz',
    5:'B',
    6:'ivan',
    7:'A',
    8:'F',
    9:'C',
    10:'david',
    11:'D',
    12:'hiran'}

# Load pre-trained models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
landmark_predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
cnn_face_recognition = keras.models.load_model(args.cnn)

# Function to calculate tilt angle
def calculate_tilt_angle(landmarks):
    eye_left = (landmarks.part(36).x, landmarks.part(36).y)
    eye_right = (landmarks.part(45).x, landmarks.part(45).y)
    dY = eye_right[1] - eye_left[1]
    dX = eye_right[0] - eye_left[0]
    angle = np.degrees(np.arctan2(dY, dX))
    return angle

# Function to detect and recognize faces : Main Function
def detect_and_reco(image):
    # Detect faces using V&J
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rect = dlib.rectangle(x, y, x+w, y+h)
        landmarks = landmark_predictor(gray, rect)

        # Calculate tilt angle
        angle = calculate_tilt_angle(landmarks)

        # Rotate the image
        center = ((x + x + w) // 2, (y + y + h) // 2)
        center = (int(center[0]), int(center[1]))
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

        # Crop the face region
        cropped_face = rotated_image[y:y+h, x:x+w]
        img1 = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2GRAY)
        img2 = cv2.equalizeHist(img1)
        img3 = cv2.resize(img2, (90, 90), interpolation = cv2.INTER_LINEAR)

        # Run the CNN classifyier
        input_vector = np.ravel(img3).reshape((-1, 90, 90, 1))/255
        print(input_vector)
        print(input_vector.shape)
        result = cnn_face_recognition.predict(input_vector, verbose=0)
    

        # Add BoundingBoxes and Tags
        color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        image = cv2.rectangle(image, (x,y), (x+w, y+h), color=color, thickness=4)
        image = cv2.putText(image, str(facemapping[np.argmax(result)]), (x,y+h), color=color, fontFace=1, fontScale=2, thickness=4)
    cv2.imshow('Capture - Face detection', frame)



camera_device = args.camera
# Read the video stream
cap = cv2.VideoCapture(camera_device)
if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)

while True:
    ret, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break

    detect_and_reco(frame)

    if cv2.waitKey(10) == 27:
        break