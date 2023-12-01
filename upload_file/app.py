import time
import logging
import requests
from flask import Flask, render_template, request, send_from_directory
from PIL import Image
import gridfs
from flask_pymongo import PyMongo
import io
from flask import send_file
from bson import ObjectId


app = Flask(__name__)
app.config["MONGO_URI"] = "mongodb://localhost:27017/myDatabase"
db = PyMongo(app).db
fs = gridfs.GridFS(db)

def generate_unique_filename():
    # Use timestamp for unique filenames
    timestamp = time.time()
    uploaded_image = f'{timestamp:.0f}_uploaded.jpg'
    result_image = f'{timestamp:.0f}_result_image.jpg'
    return timestamp, uploaded_image, result_image

@app.route('/retrieve/<image_id>')
def retrieve(image_id):
    # Get the image from GridFS using the provided image_id
    image_data = fs.get(ObjectId(image_id))

    # Check if the image data exists
    if image_data:
        # Send the image file back to the user
        return send_file(io.BytesIO(image_data.read()), mimetype='image/png')
    else:
        return "Image not found"


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST' and 'image' in request.files:
        image = request.files['image']

        # Generate unique filenames using timestamp
        timestamp, uploaded_image_filename, result_image_filename = generate_unique_filename()

        # Process the image
        img = Image.open(image).convert('RGB')

        # Convert image to bytes
        img_byte_array = io.BytesIO()
        img.save(img_byte_array, format='PNG')
        img_byte_array = img_byte_array.getvalue()

        # Save uploaded image to GridFS
        uploaded_image_id = fs.put(img_byte_array, filename=f'{float(timestamp):.0f}_uploaded_image.png')

        # Send timestamp and image to model server
        model_server_url = 'http://127.0.0.1:5001/predict'  # Adjust the URL accordingly
        files = {'image': img_byte_array}
        data = {'timestamp': timestamp}

        try:
            response = requests.post(model_server_url, files=files, data=data)
            response.raise_for_status()  # Raises HTTPError if the HTTP request returned an unsuccessful status code
            result_image_path = response.json().get('result_image_id', None)
            message = "Image uploaded successfully."

        except Exception as e:
            logging.exception(f"An error occurred: {e}")
            result_image_path = None
            message = "An error occurred while processing the image."

        return render_template('index.html', uploaded_image=uploaded_image_id, result_image=result_image_path, message=message)

    return render_template('index.html', uploaded_image=None, result_image=None, message=None)

@app.route('/uploads/<filename>')
def uploaded_image(filename):
    return send_from_directory(app.root_path, filename)

if __name__ == '__main__':
    app.run(debug=True)