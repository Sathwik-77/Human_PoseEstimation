import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from pose_estimation import preprocess_image, draw_keypoints_per_person, draw_skeleton_per_person, estimate_pose

st.title("Human Pose Estimation")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Load the uploaded image
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    # Convert the image to a format suitable for OpenCV
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Preprocess the image
    img, img_tensor = preprocess_image(img_array)

    # Perform pose estimation
    output = estimate_pose(img_tensor)

    # Debug print to check the output structure
    st.write(output)

    # Ensure the keys exist in the model output
    if "keypoints" in output and "keypoints_scores" in output and "scores" in output:
        # Draw keypoints and skeleton
        keypoints_img = draw_keypoints_per_person(img, output["keypoints"], output["keypoints_scores"], output["scores"], keypoint_threshold=2)
        skeletal_img = draw_skeleton_per_person(img, output["keypoints"], output["keypoints_scores"], output["scores"], keypoint_threshold=2)

        # Convert images back to RGB for display
        keypoints_img = cv2.cvtColor(keypoints_img, cv2.COLOR_BGR2RGB)
        skeletal_img = cv2.cvtColor(skeletal_img, cv2.COLOR_BGR2RGB)

        # Display the original image
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Display the images with keypoints and skeleton
        st.image(keypoints_img, caption="Keypoints Image", use_column_width=True)
        st.image(skeletal_img, caption="Skeleton Image", use_column_width=True)

        # Save the images if needed
        keypoints_pil = Image.fromarray(keypoints_img)
        skeletal_pil = Image.fromarray(skeletal_img)
        keypoints_pil.save("output/keypoints-img.jpg")
        skeletal_pil.save("output/skeleton-img.jpg")
    else:
        st.error("Model output does not contain expected keys.")
