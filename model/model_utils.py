import os
import redis
import torch
import cv2
from PIL import Image
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.transforms import functional as F

# Get Redis host and port from environment variables or use default values
redisHost = os.getenv("REDIS_HOST") or "redis-stack" #uses the container name of the redis container
redisPort = os.getenv("REDIS_PORT") or 6379

# Create a Redis client
redisClient = redis.StrictRedis(host=redisHost, port=redisPort, db=0)

def load_model(model_path):
    """Load the RetinaNet model from the specified path."""
    model = retinanet_resnet50_fpn(weights=None)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model       

def process_prediction(image, model, detection_threshold):
    """Process the image using the RetinaNet model and filter predictions based on the detection threshold."""
    with torch.no_grad():
        outputs = model(image)

    scores = list(outputs[0]['scores'].detach().cpu().numpy())
    boxes = outputs[0]['boxes'].detach().cpu().numpy()
    labels = outputs[0]['labels'].cpu().numpy()

    # Filter predictions based on the detection threshold
    thresholded_preds_indices = [i for i, score in enumerate(scores) if score > detection_threshold]
    filtered_boxes = boxes[thresholded_preds_indices]
    filtered_classes = labels[thresholded_preds_indices]

    return filtered_boxes, filtered_classes

def draw_boxes(image, boxes, classes, coco_names, colors):
    """Draw bounding boxes on the image based on the model predictions."""
    image_np = image.copy()

    for i, box in enumerate(boxes):
        if 0 <= classes[i] < len(coco_names):
            color = colors[classes[i]]
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
