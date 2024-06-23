import streamlit as st
import numpy as np
import cv2
from PIL import Image

face_classifier = cv2.CascadeClassifier(r'C:\Users\LENOVO\Naresh IT\June\13th- intro to cv2\open cv -- practicle\Haarcascades/haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier(r'C:\Users\LENOVO\Naresh IT\June\13th- intro to cv2\open cv -- practicle\Haarcascades/haarcascade_eye.xml')
 
 
def face_eye_detection(img):
   # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
    # When no faces detected, face_classifier returns and empty tuple
    if faces is ():
       print("No Face Found")
       
    # Draw rectangles around faces and eyes
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (127,0,255), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_classifier.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (255,255,0), 2)
    
    return img


def main():
    st.set_page_config(page_title="Face and Eyes Detection App", page_icon="📷", layout="wide")
    st.title("Face and Eyes Detection App")


    # Upload image file on the left side
    st.sidebar.title("Select Mode")
    mode = st.sidebar.radio("Choose the Mode", ('Upload Image', 'Webcam'))
 
    if mode == 'Upload Image':
        uploaded_file = st.sidebar.file_uploader("Upload Image", type=['png', 'jpg'])

        if uploaded_file is not None:
            # Convert the file to an OpenCV image
            image = np.array(Image.open(uploaded_file))

            # Display original image on the left side
            st.sidebar.subheader("Original Image")
            st.sidebar.image(image, caption="Original Image", use_column_width=True)

            # Process image
            detected_img = face_eye_detection(image)

            # Resize the image to fit the frame
            resized_img = cv2.resize(detected_img, (1000, 1500))

            # Display processed image
            st.subheader("Processed Image")
            st.image(resized_img, caption="Detected Faces and Eyes", use_column_width=True)

    elif mode == "Webcam":
        st.sidebar.write("Press 'Start' to begin webcam")
        status = st.sidebar.radio("Webcam Status", ('Start', 'Stop'))
        st.sidebar.write("Press 'Stop' to end the webcam")
        if status == 'Start':

            FRAME_WINDOW = st.image([])

            video_capture = cv2.VideoCapture(0)

            while status == 'Start':
                ret, frame = video_capture.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    detected_frame = face_eye_detection(frame)
                    FRAME_WINDOW.image(detected_frame)
                else:
                    
                    break
 
            video_capture.release()  # Release video capture when done

        elif status == 'Stop':
            st.write("Webcam has been stopped")


if __name__ == "__main__":
    main()