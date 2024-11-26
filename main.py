import streamlit as st
import cv2
import numpy as np
import os
import pandas as pd
import time

# Define the input and output directories
input_folder = r'D:\Saran\Minor project\python\Scripts\moon_imgs'  # Update with your folder
output_folder = r'D:\Saran\Minor project\python\Scripts\processed_images'  # Update with your folder

# Create the output directory if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Initialize an empty list to store crater data
crater_data = []

# Function to detect pits (craters) and boulders
def detect_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (9, 9), 2)

    # HoughCircles for detecting pits (circular shapes)
    circles = cv2.HoughCircles(
        blurred_image, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
        param1=100, param2=30, minRadius=10, maxRadius=100
    )

    # Detect contours for boulders
    _, threshold_image = cv2.threshold(blurred_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output_image = image.copy()  # Keep the original image intact

    # Mark the detected pits (craters) and store their data
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            # Draw green circle around the pit
            cv2.circle(output_image, (x, y), r, (0, 255, 0), 2)
            crater_data.append({"x": x, "y": y, "radius": r, "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')})

    # Mark the detected boulders
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # Adjust area threshold as needed
            x, y, w, h = cv2.boundingRect(contour)
            # Draw green rectangle around the boulder
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return output_image

# Function to save crater list to a CSV
def save_crater_list_to_csv():
    df = pd.DataFrame(crater_data)
    if not df.empty:
        df.to_csv('crater_list.csv', index=False)
        st.success("Crater list saved successfully.")
    else:
        st.warning("No craters detected to save.")

# Dashboard navigation
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Prediction"])

# Home Page
if app_mode == "Home":
    st.header("Detection of Craters and Boulders on Lunar Surface")
    st.markdown("""
    This project detects craters (pits) and boulders on lunar surface images using computer vision techniques.
    """)

# About Page
elif app_mode == "About":
    st.header("About")
    st.markdown("""
    #### Objective:
    This project detects craters and boulders on lunar images using machine learning and image processing techniques.
    The detected craters are listed with their coordinates and sizes and can be saved into a file for analysis.
    """)

# Prediction Page for Moon Image Upload
elif app_mode == "Prediction":
    st.header("Detection of Craters and Boulders in Lunar Surface")
    test_image = st.file_uploader("Choose a Moon Image:")

    if test_image and st.button("Process Image"):
        # Load the image
        image = cv2.imdecode(np.frombuffer(test_image.read(), np.uint8), cv2.IMREAD_COLOR)
        if image is not None:
            # Process the image to detect features
            processed_image = detect_features(image)

            # Display the processed image
            st.image(processed_image, channels="BGR", use_container_width=True)
            st.success("Craters and boulders detected and marked.")

            # Show the crater list
            if crater_data:
                st.subheader("Detected Crater List:")
                crater_df = pd.DataFrame(crater_data)
                st.dataframe(crater_df)

                # Option to save crater list to CSV
                save_option = st.button("Save Crater List")
                if save_option:
                    save_crater_list_to_csv()
                    

        else:
            st.error("Error loading the image. Please ensure the file is a valid image.")
            

    st.write("Upload a moon image to detect and mark craters and boulders.")

