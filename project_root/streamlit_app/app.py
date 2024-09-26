import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from models.segmentation_model import segment_image
from models.identification_model import identify_objects
from models.text_extraction_model import extract_text
from utils.visualization import visualize_image
from utils.postprocessing import extract_objects_and_save
from PIL import Image

st.title("AI Image Segmentation and Object Analysis")

# Image upload
uploaded_image = st.file_uploader("Upload an Image", type=['png', 'jpg', 'jpeg'])

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Segmentation
    st.write("Running segmentation model...")
    segmentation_output = segment_image(image)

     # Extract objects and store
    st.write("Extracting objects and saving them...")
    save_dir = "data/segmented_objects"
    objects_metadata = extract_objects_and_save(image, segmentation_output, save_dir)

    # Display the metadata
    st.write("Object extraction completed. Metadata:")
    st.json(objects_metadata)
    
    # Visualization of segmentation
    st.write("Visualizing segmentation results...")
    visualize_image(image, segmentation_output[0]['masks'])

    # Object identification
    st.write("Identifying objects...")
    identified_objects = identify_objects(image)
    st.write("Identified objects:", identified_objects)

    # Text extraction
    st.write("Extracting text from objects...")
    text_data = extract_text(image)
    st.write("Extracted Text:", text_data)
