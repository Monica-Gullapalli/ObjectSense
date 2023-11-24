import os
import time
from flask import Flask, render_template, request, send_from_directory
from PIL import Image
import numpy as np
import torch
import cv2
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.transforms import functional as F

app = Flask(__name__)

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

UPLOAD_FOLDER = 'static/uploaded_images'
OUTPUT_FOLDER = 'static/outputs'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Create the required directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Load the model from the pickle file
def load_model(model_path):
    model = retinanet_resnet50_fpn(weights=None)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Load the model once
model_path = r'H:\DCSC\ObjectSense\model\retina_net.pkl'
model = load_model(model_path)

# Detection threshold
detection_threshold = 0.5

def generate_unique_filename():
    # Use timestamp for unique filenames
    timestamp = time.time()
    uploaded_image = f'{timestamp:.0f}_uploaded.jpg'
    result_image = f'{timestamp:.0f}_result_image.jpg'
    return uploaded_image, result_image

def process_prediction(image, model, detection_threshold):
    with torch.no_grad():
        outputs = model(image)

    scores = list(outputs[0]['scores'].detach().cpu().numpy())
    boxes = outputs[0]['boxes'].detach().cpu().numpy()
    labels = outputs[0]['labels'].cpu().numpy()

    thresholded_preds_indices = [i for i, score in enumerate(scores) if score > detection_threshold]
    filtered_boxes = boxes[thresholded_preds_indices]
    filtered_classes = labels[thresholded_preds_indices]

    return filtered_boxes, filtered_classes

def draw_boxes(image, boxes, classes, coco_names):
    image_np = np.array(image)

    for i, box in enumerate(boxes):
        if 0 <= classes[i] < len(coco_names):
            color = COLORS[classes[i]]
            cv2.rectangle(
                image_np,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                color, 2
            )
            cv2.putText(image_np, coco_names[classes[i]], (int(box[0]), int(box[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                        lineType=cv2.LINE_AA)
        else:
            print(f"Invalid class index: {classes[i]}")

    return Image.fromarray(image_np)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST' and 'image' in request.files:
        image = request.files['image']

        # Generate unique filenames using timestamp
        uploaded_image_filename, result_image_filename = generate_unique_filename()

        # Save the uploaded image
        uploaded_image_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_image_filename)
        image.save(uploaded_image_path)

        # Process the image
        img = Image.open(image).convert('RGB')
        img_tensor = F.to_tensor(img).unsqueeze(0)

        # Use the model for predictions
        with torch.no_grad():
            boxes, pred_classes = process_prediction(img_tensor, model, detection_threshold)

        # Process the prediction to draw bounding boxes on the image
        result_image = draw_boxes(img, boxes, pred_classes, coco_names)
        
        # Save the result image
        result_image_path = os.path.join(app.config['OUTPUT_FOLDER'], result_image_filename)
        result_image.save(result_image_path)

        return render_template('index.html', uploaded_image=uploaded_image_filename, result_image=result_image_filename)

    return render_template('index.html', uploaded_image=None, result_image=None)

@app.route('/uploads/<filename>')
def uploaded_image(filename):
    return send_from_directory(app.root_path, filename)

if __name__ == '__main__':
    app.run(debug=True)
