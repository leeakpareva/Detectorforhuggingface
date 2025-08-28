from ultralytics import YOLO # type: ignore
import cv2
import numpy as np

# Load a pre-trained YOLOv8 model (nano version = small & fast)
model = YOLO("yolov8n.pt")

def detect_objects(image):
    """
    Run YOLO on the input image.
    Returns:
      - annotated image with bounding boxes
      - list of detected object names
    """
    # Handle different image formats and channel counts
    if isinstance(image, np.ndarray):
        # If image has 4 channels (RGBA), convert to RGB
        if image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        # If image has 1 channel (grayscale), convert to RGB
        elif len(image.shape) == 2 or image.shape[-1] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    results = model(image)
    annotated_img = results[0].plot()

    # Extract detected object names
    detected_objects = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0].item())  # class ID
        label = results[0].names[cls_id]  # class name
        detected_objects.append(label)

    return annotated_img, detected_objects
