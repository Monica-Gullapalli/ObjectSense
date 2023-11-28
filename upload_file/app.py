import os
import time
import requests
from flask import Flask, render_template, request, send_from_directory

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploaded_images'
OUTPUT_FOLDER = 'static/outputs'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Create the required directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

def generate_unique_filename():
    # Use timestamp for unique filenames
    timestamp = time.time()
    uploaded_image = f'{timestamp:.0f}_uploaded.jpg'
    result_image = f'{timestamp:.0f}_result_image.jpg'
    return timestamp, uploaded_image, result_image

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST' and 'image' in request.files:
        image = request.files['image']

        # Generate unique filenames using timestamp
        timestamp, uploaded_image_filename, result_image_filename = generate_unique_filename()

        # Save the uploaded image
        uploaded_image_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_image_filename)
        image.save(uploaded_image_path)

        # Send timestamp and image to model server
        model_server_url = 'http://127.0.0.1:5001/predict'  # Adjust the URL accordingly
        files = {'image': open(uploaded_image_path, 'rb')}
        data = {'timestamp': timestamp}
        response = requests.post(model_server_url, files=files, data=data)

        # Get the result image path from the response
        try:
            result_image_path = response.json().get('result_image', None)
        except requests.exceptions.JSONDecodeError as e:
            print(f"Error decoding JSON response: {e}")
            result_image_path = None

        return render_template('index.html', uploaded_image=uploaded_image_filename, result_image=result_image_path)

    return render_template('index.html', uploaded_image=None, result_image=None)

@app.route('/uploads/<filename>')
def uploaded_image(filename):
    return send_from_directory(app.root_path, filename)

if __name__ == '__main__':
    app.run(debug=True)