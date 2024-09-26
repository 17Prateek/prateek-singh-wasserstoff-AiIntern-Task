import torch
from torchvision import models, transforms


def load_segmentation_model():
    # Load the pre-trained Mask R-CNN model
    model = models.detection.maskrcnn_resnet50_fpn(weights="COCO_V1")
    model.eval()
    return model

def segment_image(image):
    # If image is already a PIL.Image object, skip loading
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image).unsqueeze(0)
    

    model = load_segmentation_model()
    
    # Perform segmentation
    with torch.no_grad():
        output = model(image_tensor)
    
    return output

