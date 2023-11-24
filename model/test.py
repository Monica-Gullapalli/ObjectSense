import torch
import torchvision
import cv2
import detect_utils
from PIL import Image
import numpy as np
import argparse

def load_model(model_path):
    # Load the model architecture
    model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=False)
    
    # Load the model weights from the pickle file
    model.load_state_dict(torch.load(model_path))
    
    # Set the model to evaluation mode
    model.eval()
    
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='Path to input image file')
    parser.add_argument('-m', '--model', help='Path to the saved model pickle file')
    parser.add_argument('-t', '--threshold', default=0.5, type=float,
                        help='Minimum confidence score for detection')
    args = parser.parse_args()

    # Load the PyTorch model from the pickle file
    model = load_model(r'H:\DCSC\ObjectSense\models\retina_net.pkl')

    # Load the input image
    image = Image.open(args.input).convert('RGB')
    image_array = np.array(image)
    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    # Perform predictions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    boxes, classes = detect_utils.predict(image, model, device, args.threshold)
    result = detect_utils.draw_boxes(boxes, classes, image_array)

    # Display the result
    cv2.imshow('Prediction Result', result)
    cv2.waitKey(0)

    # Save the result in the same directory
    result_path = f"{args.input.split('/')[-1].split('.')[0]}_prediction.jpg"
    cv2.imwrite(result_path, result)
    print(f"Prediction result saved at: {result_path}")

if __name__ == "__main__":
    main()