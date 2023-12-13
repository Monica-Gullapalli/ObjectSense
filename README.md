# ObjectSense

ObjectSense is a web application that allows users to upload images for processing by a machine learning model. It utilizes Flask for the web interface, MongoDB for image storage, Redis for communication between components, and communicates with a separate model server for image processing.

# Team Members:
- Yash Panchal
- Aman Sheth
- Monica Gullapalli

## Table of Contents
1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Working](#working)
4. [Usage](#usage)

## Requirements
Make sure you have the following dependencies installed:
- opencv-python==4.8.1.78
- torchvision==0.16.1
- Flask==3.0.0
- Pillow==10.1.0
- pickle-mixin==1.0.2
- flask-cors==4.0.0
- pymongo==4.6.0
- Jinja2==3.1.2
- Flask-PyMongo==2.3.0
- jsonpickle==3.0.2
- redis==5.0.1

## Installation
- Install required libraries

     ` pip install -r requirements.txt`

## Working

The `app.py` script serves as the main component of the ObjectSense web application.

### 1. Initialization and Configuration

The script initializes the Flask application, configures the MongoDB connection, and sets up GridFS for image storage. Additionally, it establishes a connection to a Redis server for logging and communication.

### 2. Logging and Image Processing

Two functions, `log` and `send_to_model`, are defined for logging messages and preparing image data for processing by the model server, respectively.

### 3. Routes

#### 3.1. `'/retrieve/<image_id>'`

This route retrieves an image from MongoDB's GridFS based on the provided image ID and returns it as a response.

#### 3.2. `'/'`

The root route handles both GET and POST requests. It processes uploaded images, converts them to bytes, saves them to GridFS, and communicates with a model server for image processing. Results are displayed on the web interface.

#### 3.3. `'/uploads/<filename>'`

This route serves static files, specifically uploaded images, from the application's root directory.


The `model_server.py` script defines a Flask application for a model server that processes images using a pre-trained object detection model.

### 1. Initialization and Configuration

- Initializes the Flask application and configures MongoDB connection using Flask-PyMongo.
- Defines the model configurations, including class names and colors.
- Loads the pre-trained RetinaNet model.

### 2. Image Processing Endpoint

- Defines a `/predict` endpoint that accepts POST requests.
- Listens for image IDs from a Redis queue (`"queue"`) using `blpop`.
- Retrieves the image from MongoDB's GridFS based on the received image ID.
- Processes the image using the loaded RetinaNet model to predict bounding boxes and classes.
- Draws bounding boxes on the image and saves the result image back to MongoDB's GridFS.
- Returns the ID of the result image.

### 3. Model Utility Functions (`model_utils.py`)

- Loads the pre-trained RetinaNet model from a specified path.
- Defines a function (`process_prediction`) for processing model predictions.
- Defines a function (`draw_boxes`) for drawing bounding boxes on the image based on predictions.

### 4. Redis Communication

- Establishes a connection to a Redis server for communication between the model server and other components.
- Listens for image IDs from the Redis queue (`"queue"`).

## Usage

To run the ObjectSense application, follow the steps below. Ensure that you have Docker installed on your machine.

1. **Run MongoDB:**

   #### Using docker:
   ```bash
   docker run --name mongo -p 27017:27017 --net test-nw -d mongodb/mongodb-community-server:latest
   ```
   #### Using kubernetes:
   ```bash
   kubectl apply -f mongo-deployment.yaml 
   ```
   ```bash
   kubectl apply -f mongo-service.yaml 
   ```
   ```bash
   kubectl port-forward --address localhost service/mongo 27017:27017
   ```

   It uses the public image available for MongoDB. This command launches a MongoDB container using the latest version. It will download the image if it is not locally available on your machine. In order to connect to the database, use the port forwarding command.

2. **Run Redis:**

   #### Using docker:
   ```bash
   docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 --net test-nw redis/redis-stack:latest
   ```
   #### Using kubernetes:
   ```bash
   kubectl apply -f redis-deployment.yaml 
   ```
   ```bash
   kubectl apply -f redis-service.yaml 
   ```

   I have used the public image available for Redis. Start a Redis container with the name `redis-stack`.

3. **Run the Model Server:**

   #### Using docker:
   ```bash
   docker run --name model_server -p 5001:5001 --net test-nw yashy2k/model_server:v1
   ```
   #### Using kubernetes:
   ```bash
   kubectl apply -f model-deployment.yaml 
   ```
   ```bash
   kubectl apply -f model-service.yaml 
   ```

   This will pull the image I built and pushed to the docker hub and will launch the Model Server container, exposing port 5001.

4. **Run the Upload File (Flask) Server:**

   #### Using docker:
   ```bash
   docker run --name upload_file -p 5000:5000 --net test-nw yashy2k/upload_file:v1
   ```
   #### Using kubernetes:
   ```bash
   kubectl apply -f app-deployment.yaml 
   ```
   ```bash
   kubectl apply -f app-service.yaml 
   ```

   The image will be pulled from the docker hub and start the Upload File container, exposing port 5000.

6. **Access the ObjectSense Web Application:**

   Open your web browser and go to [http://localhost:5000](http://localhost:5000) or find the port given by kubenetes. This is the main interface for uploading and processing images. 

7. **Upload an Image:**

   - Click on the "Choose File" button to select an image.
   - Click the "Upload" button to submit the image for processing.

   The application will process the image using the pre-trained RetinaNet model through the Model Server.

8. **View Results:**

   - The uploaded image and the result image with bounding boxes will be displayed on the web interface.

   **Note:** Ensure that the Docker containers are running, and ports are correctly mapped. If you encounter any issues, refer to the Docker logs for each container.