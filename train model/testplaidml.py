import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from PIL import Image
import numpy as np

# Input resolution of training images
img_width, img_height = 300, 300

# Data augmentation and rescaling
trainRescale = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # Split data into training and validation sets
)

# Flow from directory will load images and label them based on the folder they are in
trainData = trainRescale.flow_from_directory(
    'Train/',
    target_size=(img_width, img_height),
    batch_size=8,
    class_mode='categorical',
    subset='training'
)

validationData = trainRescale.flow_from_directory(
    'Train/',
    target_size=(img_width, img_height),
    batch_size=8,
    class_mode='categorical',
    subset='validation'
)

# Load pre-trained VGG16 model + higher level layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Define model
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(trainData.class_indices), activation='softmax')
])

# Compile model
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=1e-4),
              metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_instrumentClassifier.keras', save_best_only=True, monitor='val_loss')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

# Train model
model.fit(
    trainData,
    steps_per_epoch=trainData.samples // trainData.batch_size,
    validation_data=validationData,
    validation_steps=validationData.samples // validationData.batch_size,
    epochs=50,
    callbacks=[early_stopping, model_checkpoint, reduce_lr]
)

# Save final model
model.save_weights('instrumentClassifier.weights.keras')
model.save('instrumentClassifier.keras')

# Load the best model
model = load_model('best_instrumentClassifier.keras')

# Get list of all images in directory
testImages = os.listdir('Test/')

# Loop through test images
for image in testImages:
    img = Image.open('Test/' + image)
    img = img.resize((img_width, img_height))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Rescale pixel values

    # Use model to produce probability of image
    prediction = model.predict(img)

    # Output prediction
    predicted_class = np.argmax(prediction, axis=1)
    class_labels = list(trainData.class_indices.keys())
    print(f"Image: {image}, Predicted Class: {class_labels[predicted_class[0]]}, Confidence: {np.max(prediction):.2f}")