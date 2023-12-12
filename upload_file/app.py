import os
import time
import logging
import requests
from flask import Flask, render_template, request, send_from_directory, send_file
from PIL import Image
import gridfs
from flask_pymongo import PyMongo
import io
from bson import ObjectId
import redis

app = Flask(__name__)
app.config["MONGO_URI"] = "mongodb://mongo:27017/myDatabase" #uses the container name of the mongodb container
db = PyMongo(app).db
fs = gridfs.GridFS(db)
redis_host = os.getenv("REDIS_HOST") or "redis-stack" #uses the container name of the redis container
redis_client = redis.Redis(host=redis_host, decode_responses=True)


def log(message):
    """Log a message to Redis."""
    redis_client.lpush("logging", message)


def send_to_model(img):
    """Send the image data to the model server via Redis queue."""
    redis_client.lpush('queue', img)


@app.route('/retrieve/<image_id>')
def retrieve(image_id):
    """Retrieve an image from GridFS using its ID."""
    image_data = fs.get(ObjectId(image_id))
    if image_data:
        return send_file(io.BytesIO(image_data.read()), mimetype='image/png')
    else:
        return "Image not found"


@app.route('/', methods=['GET', 'POST'])
def index():
    """Handle the main page, including image processing and interaction with the model server."""
    if request.method == 'POST':
        if 'image' not in request.files:
            # No image selected
            return render_template('index.html', uploaded_image=None, result_image=None,
                                   message="Please select an image to upload.")

        image = request.files['image']
        if image.filename == '':
            # Empty file name indicates no file selected
            return render_template('index.html', uploaded_image=None, result_image=None,
                                   message="Please select an image to upload.")

        timestamp = time.time()

        # Process the image
        img = Image.open(image).convert('RGB')

        # Convert image to bytes
        img_byte_array = io.BytesIO()
        img.save(img_byte_array, format='PNG')
        img_byte_array = img_byte_array.getvalue()

        # Save uploaded image to GridFS
        uploaded_image_id = fs.put(img_byte_array, filename=f'{int(timestamp)}_uploaded_image.png')

        # Send timestamp and image to the model server
        model_server_url = 'http://model_server:5001/predict'  # Adjust the URL accordingly
        data = {'timestamp': timestamp}

        # Push to the Redis queue
        send_to_model(img=str(uploaded_image_id))

        try:
            response = requests.post(model_server_url, data=data, timeout=120)
            response.raise_for_status()
            result_image_path = response.json().get('result_image_id', None)
            message = "Image processed successfully."

        except Exception as e:
            logging.exception(f"An error occurred: {e}")
            result_image_path = None
            message = "An error occurred while processing the image."

        return render_template('index.html', uploaded_image=uploaded_image_id, result_image=result_image_path,
                               message=message)

    return render_template('index.html', uploaded_image=None, result_image=None, message=None)


@app.route('/uploads/<filename>')
def uploaded_image(filename):
    """Serve uploaded images from the server."""
    return send_from_directory(app.root_path, filename)


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)
