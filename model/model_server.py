import os
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
import torch
import numpy as np
from model_utils import *
from torchvision.transforms import functional as F
import gridfs
from flask_pymongo import PyMongo
import io
from bson import ObjectId

app = Flask(__name__)
CORS(app)
app.config["MONGO_URI"] = "mongodb://mongo:27017/myDatabase"
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

# Load the model once
model_path = os.path.join(os.getcwd(), 'retina_net.pkl')
model = load_model(model_path)

# Detection threshold
detection_threshold = 0.5

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # if 'image' not in request.files:
        #     return jsonify({'error': 'No image provided'})

        # image = request.files['image']
        work = redisClient.blpop("queue", timeout=0)
        if work:
            img1 = work[1].decode('utf-8')
            image_data = fs.get(ObjectId(img1))
            image = image_data.read()
            
            # Process the image
            img = Image.open(io.BytesIO(image)).convert('RGB')
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
                result_image_id = fs.put(result_img_byte_array, filename=f'{float(timestamp):.0f}_result_image.png')
                return jsonify({'result_image_id': str(result_image_id)})
                
            else:
                return jsonify({'error': 'No timestamp provided'})
        
    except Exception as e:
        app.logger.error(f"An error occurred: {e}")
        return jsonify({'error': 'An internal server error occurred'}), 500

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5001, threaded=True)