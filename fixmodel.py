from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D

# Custom DepthwiseConv2D class to ignore 'groups' argument
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop('groups', None)  # Remove 'groups' if it exists
        super().__init__(*args, **kwargs)

# Load the model with the custom layer
print("Loading model...")
model = load_model("model/keras_model.h5", custom_objects={"DepthwiseConv2D": CustomDepthwiseConv2D})
print("Model loaded successfully!")

# Print model summary
model.summary()

# Save the fixed model
model.save("model/fixed_keras_model.h5")
print("Model saved successfully as fixed_keras_model.h5")
