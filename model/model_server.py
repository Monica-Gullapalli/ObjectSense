import os
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from PIL import Image
import torch
import numpy as np
from model_utils import load_model, process_prediction, draw_boxes
from torchvision.transforms import functional as F
import gridfs
from flask_pymongo import PyMongo
import io
from flask import send_file

app = Flask(__name__)
CORS(app)
app.config["MONGO_URI"] = "mongodb://localhost:27017/myDatabase"
db = PyMongo(app).db
fs = gridfs.GridFS(db)

# Model configurations
coco_names = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'Camera'
]

COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))

UPLOAD_FOLDER = r'E:\DCSC_project\ObjectSense\upload_file\static\uploaded_images'
OUTPUT_FOLDER = r'E:\DCSC_project\ObjectSense\upload_file\static\outputs'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Load the model once
model_path = r'E:\DCSC_project\ObjectSense\model\retina-model.pkl'
model = load_model(model_path)

# Detection threshold
detection_threshold = 0.5

# @app.route('/retrieve/<image_id>')
# def retrieve(image_id):
#     # Get the image from GridFS using the provided image_id
#     image_data = fs.get(ObjectId(image_id))

#     # Check if the image data exists
#     if image_data:
#         # Send the image file back to the user
#         return send_file(io.BytesIO(image_data.read()), mimetype='image/png')
#     else:
#         return "Image not found"

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'})

    image = request.files['image']

    # Process the image
    img = Image.open(image).convert('RGB')
    img_tensor = F.to_tensor(img).unsqueeze(0)

    # Use the model for predictions
    with torch.no_grad():
        boxes, pred_classes = process_prediction(img_tensor, model, detection_threshold)

    # Process the prediction to draw bounding boxes on the image
    result_image = draw_boxes(np.array(img), boxes, pred_classes, coco_names, COLORS)

    # Save the result image with the received timestamp
    timestamp = request.form.get('timestamp', None)
    if timestamp:
        # Convert result image to bytes
        result_img_byte_array = io.BytesIO()
        result_image.save(result_img_byte_array, format='PNG')
        result_img_byte_array = result_img_byte_array.getvalue()

        # Save result image to MongoDB using GridFS
        result_image_filename = f'{float(timestamp):.0f}_result_image.jpg'
        result_image_id = fs.put(result_img_byte_array, filename=f'{float(timestamp):.0f}_result_image.png')
        return jsonify({'result_image_id': str(result_image_id)})
        
    else:
        return jsonify({'error': 'No timestamp provided'})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
