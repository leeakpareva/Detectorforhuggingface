"""
Enhanced YOLO detection with improved accuracy, color detection, and detailed attributes
"""
from ultralytics import YOLO # type: ignore
import cv2 # type: ignore
import numpy as np # type: ignore
from collections import Counter
import webcolors
from sklearn.cluster import KMeans # type: ignore
import torch # type: ignore

# Load a more accurate YOLO model
# For better accuracy, use yolov8m.pt or yolov8l.pt instead of yolov8n.pt
model_size = 'yolov8m.pt'  # Medium model for better accuracy vs speed balance
model = YOLO(model_size)

# Set higher confidence threshold for better accuracy
CONFIDENCE_THRESHOLD = 0.5  # Increase this for fewer but more accurate detections
NMS_THRESHOLD = 0.45  # Non-maximum suppression threshold

def get_dominant_colors(image, n_colors=3):
    """
    Extract dominant colors from an image region using K-means clustering
    """
    try:
        # Reshape image to be a list of pixels
        pixels = image.reshape((-1, 3))
        
        # Apply K-means to find dominant colors
        kmeans = KMeans(n_clusters=min(n_colors, len(pixels)), random_state=42)
        kmeans.fit(pixels)
        
        # Get the colors
        colors = kmeans.cluster_centers_.astype(int)
        
        # Get color names
        color_names = []
        for color in colors:
            try:
                # Find closest named color
                closest_name = get_color_name(color)
                color_names.append(closest_name)
            except:
                color_names.append(f"RGB({color[0]},{color[1]},{color[2]})")
        
        return color_names
    except:
        return ["Unknown"]

def get_color_name(rgb_color):
    """
    Convert RGB values to a human-readable color name
    """
    min_colors = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - rgb_color[0]) ** 2
        gd = (g_c - rgb_color[1]) ** 2
        bd = (b_c - rgb_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]

def analyze_object_attributes(image, box, label):
    """
    Analyze detailed attributes of detected objects
    """
    x1, y1, x2, y2 = box
    object_region = image[int(y1):int(y2), int(x1):int(x2)]
    
    attributes = {
        'label': label,
        'position': get_position_description(x1, y1, x2, y2, image.shape),
        'size': get_size_description(x2-x1, y2-y1, image.shape),
        'colors': get_dominant_colors(object_region, n_colors=2),
        'confidence': None  # Will be set from detection
    }
    
    return attributes

def get_position_description(x1, y1, x2, y2, image_shape):
    """
    Describe object position in human terms
    """
    h, w = image_shape[:2]
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    # Horizontal position
    if center_x < w / 3:
        h_pos = "left"
    elif center_x > 2 * w / 3:
        h_pos = "right"
    else:
        h_pos = "center"
    
    # Vertical position
    if center_y < h / 3:
        v_pos = "top"
    elif center_y > 2 * h / 3:
        v_pos = "bottom"
    else:
        v_pos = "middle"
    
    if h_pos == "center" and v_pos == "middle":
        return "center"
    elif v_pos == "middle":
        return h_pos
    elif h_pos == "center":
        return v_pos
    else:
        return f"{v_pos}-{h_pos}"

def get_size_description(width, height, image_shape):
    """
    Describe object size relative to image
    """
    img_area = image_shape[0] * image_shape[1]
    obj_area = width * height
    ratio = obj_area / img_area
    
    if ratio > 0.5:
        return "very large"
    elif ratio > 0.25:
        return "large"
    elif ratio > 0.1:
        return "medium"
    elif ratio > 0.05:
        return "small"
    else:
        return "tiny"

def detect_objects_enhanced(image, confidence_threshold=CONFIDENCE_THRESHOLD):
    """
    Enhanced YOLO detection with improved accuracy and detailed attributes
    Returns:
      - annotated image with bounding boxes
      - list of detected object names
      - detailed attributes for each detection
    """
    # Handle different image formats
    if isinstance(image, np.ndarray):
        if image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif len(image.shape) == 2 or image.shape[-1] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Run YOLO with custom parameters for better accuracy
    results = model(
        image,
        conf=confidence_threshold,  # Confidence threshold
        iou=NMS_THRESHOLD,  # NMS IoU threshold
        imgsz=640,  # Image size (can increase for better accuracy)
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Get annotated image
    annotated_img = results[0].plot(
        conf=True,  # Show confidence scores
        line_width=2,
        font_size=10
    )
    
    # Extract detailed information
    detected_objects = []
    detailed_attributes = []
    
    for box in results[0].boxes:
        if box.conf[0] >= confidence_threshold:  # Double-check confidence
            cls_id = int(box.cls[0].item())
            label = results[0].names[cls_id]
            confidence = float(box.conf[0].item())
            
            # Get box coordinates
            xyxy = box.xyxy[0].tolist()
            
            # Analyze attributes
            attributes = analyze_object_attributes(image, xyxy, label)
            attributes['confidence'] = f"{confidence:.2%}"
            
            detected_objects.append(label)
            detailed_attributes.append(attributes)
    
    return annotated_img, detected_objects, detailed_attributes

def get_intelligence_report(detailed_attributes):
    """
    Generate an intelligent report about detected objects
    """
    if not detailed_attributes:
        return "No objects detected in the image."
    
    report = []
    report.append(f"Detected {len(detailed_attributes)} object(s):")
    
    for attr in detailed_attributes:
        colors_str = " and ".join(attr['colors'][:2]) if attr['colors'] else "unknown colors"
        report.append(
            f"- A {attr['size']} {colors_str} {attr['label']} "
            f"in the {attr['position']} of the image "
            f"(confidence: {attr['confidence']})"
        )
    
    # Add summary statistics
    object_types = Counter([attr['label'] for attr in detailed_attributes])
    if len(object_types) > 1:
        report.append("\nSummary:")
        for obj_type, count in object_types.most_common():
            report.append(f"  • {count} {obj_type}(s)")
    
    return "\n".join(report)

# Backward compatibility wrapper
def detect_objects(image):
    """
    Wrapper for backward compatibility with original function
    """
    annotated_img, detected_objects, _ = detect_objects_enhanced(image)
    return annotated_img, detected_objects