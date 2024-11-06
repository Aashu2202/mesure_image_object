import cv2
import numpy as np
from flask import Flask, request, jsonify
import torch
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Load YOLOv5 model (Pre-trained)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Load the YOLOv5 small model

def decode_base64_to_image(base64_str):
    """Convert base64 string to image."""
    img_data = base64.b64decode(base64_str)
    img = Image.open(BytesIO(img_data))
    return np.array(img)

def calculate_object_dimensions(image, depth=0.5):
    """Calculate realistic dimensions of the object using YOLO."""
    
    # Run YOLO model to detect objects
    results = model(image)
    detections = results.pandas().xywh[0]  # Get detection results as pandas DataFrame

    if detections.empty:
        return None  # No objects detected

    # Apply confidence threshold
    detections = detections[detections['confidence'] >= 0.2]
    if detections.empty:
        return None  # No objects meet confidence threshold

    # Assume first detected object for simplicity
    object_data = detections.iloc[0]

    # Extract bounding box dimensions
    x_center, y_center, width, height = object_data[['xcenter', 'ycenter', 'width', 'height']]
    
    # Convert relative dimensions to absolute pixel values
    img_height, img_width, _ = image.shape
    obj_width_pixels = width
    obj_height_pixels = height

    # Scaling adjustments (adjust based on actual scale, e.g., real-world measurements)
    scale_factor = 0.01  # Adjusted scale factor for more realistic measurements
    object_width_m = obj_width_pixels * scale_factor * depth
    object_height_m = obj_height_pixels * scale_factor * depth

    # Convert to various units
    def to_units(val_m):
        val_cm = val_m * 100
        val_in = val_cm * 0.393701
        val_ft = val_in / 12
        return {
            'meter': round(val_m, 2),
            'cm': round(val_cm, 2),
            'inch': round(val_in, 2),
            'feet': round(val_ft, 2),
        }

    return {
        'length': to_units(object_height_m),
        'width': to_units(object_width_m),
        'area': {
            'cm2': round(object_width_m * object_height_m * 10000, 2),
            'inch2': round((object_width_m * object_height_m) * 1550, 2),
            'ft2': round((object_width_m * object_height_m) * 10.7639, 2),
            'm2': round(object_width_m * object_height_m, 2)
        }
    }
@app.route('/measure-object', methods=['POST'])
def measure_object():
    """Handle the POST request to measure object dimensions."""
    data = request.json
    base64_image = data.get('image')
    depth = data.get('depth', 0.5)  # Default depth to 0.5 meters if not provided

    # Decode the image from base64
    image = decode_base64_to_image(base64_image)

    # Calculate object dimensions using YOLO and the assumed depth
    dimensions = calculate_object_dimensions(image, depth)
    
    if not dimensions:
        return jsonify({'error': 'Object not detected'}), 400

    return jsonify({
        'dimensions': dimensions
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
