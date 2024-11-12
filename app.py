import cv2
import numpy as np
import base64
from flask_cors import CORS
from flask import Flask, request, jsonify

app = Flask(__name__)

CORS(app)
# Function to decode the Base64 image
def decode_image(base64_string):
    try:
        # Remove any data URL prefix if it exists
        if "base64," in base64_string:
            base64_string = base64_string.split("base64,")[1]

        # Add padding if necessary for Base64 decoding
        while len(base64_string) % 4 != 0:
            base64_string += '='

        # Decode the base64 image
        img_data = base64.b64decode(base64_string)
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)

        # Check if image was decoded successfully
        if img is None:
            raise ValueError("Image decoding failed. The provided data may not be a valid image.")
        
        return img
    except Exception as e:
        raise ValueError(f"Error decoding image: {str(e)}")

# Function to preprocess the image
def preprocess_image(img, thresh_1=57, thresh_2=232):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
    img_canny = cv2.Canny(img_blur, thresh_1, thresh_2)

    kernel = np.ones((3, 3), np.uint8)
    img_dilated = cv2.dilate(img_canny, kernel, iterations=1)
    img_closed = cv2.morphologyEx(img_dilated, cv2.MORPH_CLOSE, kernel, iterations=4)

    return img_closed

# Function to find contours in the image
def find_contours(img_preprocessed):
    contours, _ = cv2.findContours(image=img_preprocessed, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    polygons = []

    for contour in contours:
        epsilon = 0.04 * cv2.arcLength(curve=contour, closed=True)
        polygon = cv2.approxPolyDP(curve=contour, epsilon=epsilon, closed=True)
        
        if polygon.shape[0] == 4:  # Ensure we only work with quadrilaterals
            polygon = polygon.reshape(4, 2)
            polygons.append(polygon)

    return polygons

# Function to reorder the coordinates of the detected polygon
def reorder_coords(polygon):
    rect_coords = np.zeros((4, 2))
    add = polygon.sum(axis=1)
    rect_coords[0] = polygon[np.argmin(add)]    # Top left
    rect_coords[3] = polygon[np.argmax(add)]    # Bottom right

    subtract = np.diff(polygon, axis=1)
    rect_coords[1] = polygon[np.argmin(subtract)]    # Top right
    rect_coords[2] = polygon[np.argmax(subtract)]    # Bottom left

    return rect_coords

# Function to calculate sizes based on polygons
def calculate_sizes(polygons, reference_length=None):
    sizes = []
    reference_scale = None
    
    for polygon in polygons:
        rect_coords = np.float32(reorder_coords(polygon))
        height = cv2.norm(rect_coords[0], rect_coords[2], cv2.NORM_L2)
        width = cv2.norm(rect_coords[0], rect_coords[1], cv2.NORM_L2)

        # If reference length is provided, calculate scale
        if reference_length:
            ref_size = max(height, width)
            reference_scale = reference_length / ref_size
            height, width = height * reference_scale, width * reference_scale

        sizes.append((height, width))

    return np.array(sizes), reference_scale

# Function to convert pixel sizes to centimeters using the provided conversion factor
def convert_to_cm(sizes_pixel, reference_scale=None):
    # If reference scale is not provided, use default pixel-to-cm conversion
    pixel_to_cm = 37.795275591 if reference_scale is None else 1 / reference_scale
    sizes_cm = [(height / pixel_to_cm, width / pixel_to_cm) for (height, width) in sizes_pixel]
    return np.array(sizes_cm)

# Endpoint to measure size
@app.route('/measure-size', methods=['POST'])
def measure_size():
    data = request.get_json()
    base64_image = data.get('image')
    reference_length = data.get('reference_length')  # Length of reference object in cm, if available

    if not base64_image:
        return jsonify({'error': 'No image provided'}), 400

    try:
        img_original = decode_image(base64_image)
        img_preprocessed = preprocess_image(img_original)
        polygons = find_contours(img_preprocessed)

        if not polygons:
            return jsonify({'error': 'No quadrilateral contours found.'}), 404

        # Calculate sizes and retrieve scale if reference_length is provided
        sizes_pixel, reference_scale = calculate_sizes(polygons, reference_length)
        sizes_cm = convert_to_cm(sizes_pixel, reference_scale)

        response = {'sizes': sizes_cm.tolist()}
        return jsonify(response), 200
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)