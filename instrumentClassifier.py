from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array


from PIL import Image
import numpy as np
import os

# Input resolution of training images
img_width, img_height = 300, 300

# Rescale pixel values of Training images from [0, 255] to [0, 1] to normal data
trainRescale = ImageDataGenerator(rescale=1./255)

# Flow from directory will load images and label them based on the folder they are in
trainData = trainRescale.flow_from_directory(
    'Train/',
    target_size=(img_width, img_height),
    batch_size=8,
    class_mode='categorical')

# Define model layers
model = Sequential()

# Conv2D 32 filters, size 3x3, low level features (edges/curves)
# ReLU Activation function (replaces negative with 0)
# MaxPooling2D, reduce image by taking max value from 2x2 region
model.add(Conv2D(32, (3, 3), input_shape=(img_width, img_height, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# repeat
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Extract more features 64 filters 3x3
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten output into 1D factor
model.add(Flatten())

# Pass to Dense layer 64 neurons
# ReLU Activation function 
# Dropout 50% of neurons to prevent overfitting/complexity
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# Dense layer with number of neurons equal to number of classes
# Softmax Activation function for multi-class classification
num_classes = len(trainData.class_indices)
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# train model, steps_per_epoch is number of batches per epoch, epochs is number of iterations
model.fit(
    trainData,
    steps_per_epoch=8,
    epochs=128)

# Save model
model.save_weights('model_weights.h5')
model.save('instrumentClassifier.h5')

# Get list of all images in directory
testImages = os.listdir('Test/')

# loop through test images
for image in testImages:
    img = Image.open('Test/' + image)
    img = img.resize((img_width, img_height))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)

    # use model to produce probability of image
    prediction = model.predict(img)

    # output prediction
    predicted_class = np.argmax(prediction, axis=1)
    class_labels = list(trainData.class_indices.keys())
    print(image, class_labels[predicted_class[0]])