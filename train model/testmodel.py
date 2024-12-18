import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

# Input resolution of images (both training and testing images should match this)
img_width, img_height = 300, 300

# Load the trained model from the .h5 file
model = load_model('instrumentClassifier.h5')

# Define the class labels (match this with the labels used during training)
class_labels = ['class1', 'class2', 'class3']  # Replace with actual class labels

# Function to process a single image and predict
def predict_image(image_path):
    img = Image.open(image_path)  # Open the image
    img = img.resize((img_width, img_height))  # Resize the image to match the model input size
    img = img_to_array(img)  # Convert image to numpy array
    img = np.expand_dims(img, axis=0)  # Add batch dimension (model expects a batch of images)
    
    img = img / 255.0  # Rescale pixel values to [0, 1], same as during training

    # Make a prediction
    prediction = model.predict(img)
    
    # Get the class with the highest probability
    predicted_class = np.argmax(prediction, axis=1)
    
    # Return the predicted class label
    return class_labels[predicted_class[0]]

# Test prediction on individual images in the 'Test' directory
test_images = os.listdir('Test/')  # Get list of all images in 'Test/' folder

for image_name in test_images:
    image_path = os.path.join('Test', image_name)  # Get full path of the image
    predicted_class = predict_image(image_path)
    print(f'{image_name}: {predicted_class}')

# ---------------------------
# Optionally, evaluate the model on a test dataset (if you have a labeled test set)
# ---------------------------

# Create a test data generator (similar to training data generator, but for the test set)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
testRescale = ImageDataGenerator(rescale=1./255)  # Rescale pixel values to [0, 1]
testData = testRescale.flow_from_directory(
    'Test/',  # Directory with test data
    target_size=(img_width, img_height),  # Resize images
    batch_size=8,  # Set batch size for evaluation
    class_mode='categorical'  # Since it's a multi-class classification task
)

# Evaluate the model on the test data
loss, accuracy = model.evaluate(testData, steps=testData.samples // testData.batch_size)
print(f'Test accuracy: {accuracy * 100:.2f}%')
