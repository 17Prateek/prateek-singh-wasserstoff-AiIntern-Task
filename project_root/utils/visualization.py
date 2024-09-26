import matplotlib.pyplot as plt
import io
import numpy as np
from PIL import Image
import streamlit as st

def visualize_image(image, segmentation_masks):
    fig, ax = plt.subplots()
    ax.imshow(image)  # Show the original image
    
    # Loop through the masks and plot them one by one
    for mask in segmentation_masks:
        mask_2d = mask[0].cpu().numpy()  # Convert to NumPy and squeeze the mask
        ax.imshow(mask_2d, alpha=0.5, cmap='jet')  # Overlay the mask
    
    # Save the figure to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Use Streamlit to display the image
    st.image(buf, caption="Segmented Image", use_column_width=True)
