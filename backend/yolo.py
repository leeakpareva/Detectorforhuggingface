from ultralytics import YOLO

# Load a pre-trained YOLOv8 model (nano version = small & fast)
model = YOLO("yolov8n.pt")

def detect_objects(image):
    """
    Run YOLO on the input image.
    Returns:
      - annotated image with bounding boxes
      - list of detected object names
    """
    results = model(image)
    annotated_img = results[0].plot()

    # Extract detected object names
    detected_objects = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0].item())  # class ID
        label = results[0].names[cls_id]  # class name
        detected_objects.append(label)

    return annotated_img, detected_objects
