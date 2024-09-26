import easyocr
import numpy as np

def extract_text(image):
    # Initialize the EasyOCR reader
    reader = easyocr.Reader(['en'])
    
    # Convert the image to a NumPy array
    image_np = np.array(image)
    

    result = reader.readtext(image_np)
    
    # Extract the text from the result
    extracted_text = []
    for (bbox, text, prob) in result:
        extracted_text.append(text)
    
    return extracted_text
