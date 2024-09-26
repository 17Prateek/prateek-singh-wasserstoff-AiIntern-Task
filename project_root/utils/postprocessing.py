import os
import json
import numpy as np
from PIL import Image
import cv2

def extract_objects_and_save(image, segmentation_output, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    master_id = generate_master_id()
    objects_metadata = []
    
    # Loop through each mask
    for idx, mask in enumerate(segmentation_output[0]['masks']):
        mask_2d = mask[0].cpu().numpy()
        
        # Create a binary mask to extract the object
        binary_mask = (mask_2d > 0.5).astype(np.uint8)
        
        # Extract the object using the mask
        extracted_object = extract_object(image, binary_mask)
        
        # Save the extracted object with a unique ID
        object_id = f"{master_id}_{idx+1}"
        object_image_path = os.path.join(save_dir, f"{object_id}.png")
        extracted_object.save(object_image_path)

        # Store metadata for the object
        object_metadata = {
            "object_id": object_id,
            "master_id": master_id,
            "file_path": object_image_path
        }
        objects_metadata.append(object_metadata)
    
    # Save metadata as a JSON file
    metadata_file = os.path.join(save_dir, f"{master_id}_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(objects_metadata, f, indent=4)
    
    return objects_metadata

def generate_master_id():
    """Generates a unique master ID for the original image"""
    import uuid
    return str(uuid.uuid4())  

def extract_object(image, mask):
    """Extract the object from the original image using the binary mask"""
    image_np = np.array(image)  # Convert PIL image to NumPy array
    masked_object = cv2.bitwise_and(image_np, image_np, mask=mask)  # Apply the mask to the image
    
    # Convert the masked object back to PIL format
    return Image.fromarray(masked_object)
