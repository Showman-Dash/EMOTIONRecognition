import numpy as np
import pandas as pd
import streamlit as st
import cv2
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model

st.header('Emotion Recognition System!')
st.subheader('I will detect how you are feeling right now :)')

st.text('Take a picture from below.')
st.text('(Make sure you have enabled permission to allow browser to access your camera)')
st.text('(Also make sure there is only one face which not at any angle)')

model = load_model('Model (1).h5')
detect_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img_file_buffer = st.camera_input("Take a snap")
if img_file_buffer is not None:
    bytes_data = img_file_buffer.getvalue()
    image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns(2)

    with col1:
        st.text('Input image:')
        st.image(image)

    new_img = image
    coordinates = detect_face.detectMultiScale(new_img)
    if len(coordinates) == 1:
        for (x, y, w, h) in coordinates:
            cv2.rectangle(new_img, (x, y), (x + w, y + h), (0, 0, 0), 2)
            with col2:
                st.text('Detected Face:')
                st.image(new_img)

        new_img1 = image
        new_img1 = image[y:y + h, x:x + w]

        new_img1 = cv2.resize(new_img1, (48, 48))
        new_img1 = np.resize(new_img1, (1, 48, 48, 3))
        new_img1 = new_img1 / 255.0

        emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

        prediction = model.predict(new_img1)
        st.write('So, at this moment your emotions seem as follows:')
        i = 0
        for label in emotions:
            st.write("\t%s = %.2f %%" % (label, prediction[0][i] * 100))
            i = i + 1

    else:
        st.text('Face could not be detected or multiple faces are being detected, retake.')
