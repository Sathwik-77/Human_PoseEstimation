import streamlit as st
import cv2
import numpy as np
import tensorflow as tf

# Load the MoveNet model
MODEL_PATH = 'https://tfhub.dev/google/movenet/singlepose/lightning/4'
model = tf.saved_model.load(MODEL_PATH)

def draw_skeleton(image, keypoints, threshold=0.5):
    height, width, _ = image.shape
    keypoints = np.squeeze(keypoints)
    for kp in keypoints:
        if kp[2] > threshold:  # Confidence score
            x = int(kp[1] * width)
            y = int(kp[0] * height)
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
    return image

def detect_pose(image):
    input_image = tf.image.resize_with_pad(image, 192, 192)
    input_image = tf.cast(input_image, dtype=tf.int32)
    input_image = tf.expand_dims(input_image, axis=0)

    # Run model inference
    keypoints_with_scores = model.signatures['serving_default'](input_image)

    keypoints = keypoints_with_scores['output_0'].numpy()

    return keypoints

st.title("Human Pose Estimation")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect pose and draw skeleton
    keypoints = detect_pose(image)
    output_image = draw_skeleton(image.copy(), keypoints)

    # Display the output
    st.image(output_image, caption='Pose Estimation', use_column_width=True)
else:
    st.write("Please upload an image to get started.")
