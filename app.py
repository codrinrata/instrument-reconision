from flask import Flask, request, render_template, send_from_directory
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image
import numpy as np
import os
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.utils import get_custom_objects, custom_object_scope
import tensorflow as tf

# Custom DepthwiseConv2D class to fix 'groups' issue
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop('groups', None)  # Ignore 'groups' argument if present
        super().__init__(*args, **kwargs)

# Register custom layer globally
get_custom_objects()["CustomDepthwiseConv2D"] = CustomDepthwiseConv2D

# Load the models safely with custom object scope
with custom_object_scope({"CustomDepthwiseConv2D": CustomDepthwiseConv2D}):
    old_model = load_model("model/instrumentClassifier.h5")
    new_model = load_model("model/fixed_keras_model.h5")

print("Models loaded successfully:")
old_model.summary()
new_model.summary()

# Initialize Flask app
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure uploads folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Class labels
class_labels_old = ['acoustic drums', 'acoustic guitar', 'bass guitar', 'electric guitar', 'flute', 
                    'gramophone', 'harmonica', 'harp', 'piano', 'saxophone', 'sitar', 'trumpet', 'violin']
class_labels_new = ['acoustic guitar', 'acoustic drums', 'bass guitar', 'electric guitar', 'flute', 
                    'gramophone', 'harmonica', 'harp', 'piano', 'saxophone', 'sitar', 'trumpet', 'violin']

# Preprocess the image dynamically based on target size
def preprocess_image(image_path, target_size):
    img = load_img(image_path, target_size=target_size)  # Resize image
    img_array = img_to_array(img)  # Convert to array
    img_array = img_array / 255.0  # Normalize to [0, 1] range
    img_array = tf.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/', methods=['GET', 'POST'])
def index():
    image_url = None
    label = None
    selected_model_name = "new"  # Default model

    if request.method == 'POST':
        # Check if a file is uploaded
        if 'image' not in request.files:
            return "No file uploaded!", 400

        file = request.files['image']

        if file:
            # Save the uploaded image
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            image_url = f"/uploads/{filename}"

            # Check which model is selected
            selected_model_name = request.form.get('model')
            if selected_model_name == 'old':
                target_size = (300, 300)  # Old model input size
                selected_model = old_model
                class_labels = class_labels_old
            else:
                target_size = (224, 224)  # New model input size
                selected_model = new_model
                class_labels = class_labels_new

            # Preprocess the image for prediction
            img = preprocess_image(filepath, target_size=target_size)

            # Make a prediction
            predictions = selected_model.predict(img)
            predicted_class = np.argmax(predictions, axis=1)[0]

            # Get the class label
            if predicted_class < len(class_labels):
                label = class_labels[predicted_class]
            else:
                label = "Unknown"

    return render_template('index.html', image_url=image_url, label=label, model_name=selected_model_name)

if __name__ == '__main__':
    app.run(debug=True)
