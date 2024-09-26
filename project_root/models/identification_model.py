import torch
from torchvision import models, transforms


def load_identification_model():
    # Load the pre-trained Faster R-CNN model
    model = models.detection.fasterrcnn_resnet50_fpn(weights="COCO_V1")
    model.eval()
    return model

def identify_objects(image):
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image).unsqueeze(0)
    

    model = load_identification_model()
    
    # Perform object detection
    with torch.no_grad():
        output = model(image_tensor)
    
    return output
