import tensorflow as tf
import keras
from PIL import Image
import numpy as np
# # Define the input shape
# input_shape = (69, 32, 3)

# # Define the model architecture
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Rescaling(1./255, input_shape=input_shape),
#     tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D((2, 2)),
#     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D((2, 2)),
#     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])

# # Compile the model
# model.compile(
#     optimizer='adam',
#     loss='binary_crossentropy',
#     metrics=['accuracy']
# )

# # Define the image data generator for the training data
# train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
#     rescale=1./255,
#     validation_split=0.2
# )

# # Generate the training data from the directories
# train_data = train_datagen.flow_from_directory(
#     'clf-data-resized',
#     target_size=input_shape[:2],
#     batch_size=32,
#     class_mode='binary',
#     subset='training'
# )

# # Generate the validation data from the directories
# val_data = train_datagen.flow_from_directory(
#     'clf-data-resized',
#     target_size=input_shape[:2],
#     batch_size=32,
#     class_mode='binary',
#     subset='validation'
# )

# # Train the model
# model.fit(
#     train_data,
#     epochs=10,
#     validation_data=val_data
# )

# # Save the model
# model.save('parking_spot_model.h5')
model = keras.models.load_model('parking_spot_model.h5')
image_test = Image.open(r'D:\UIT\AI Project\Parking spot detection and counter\Final_Project\clf-data-resized\empty\00000000_00000164.jpg')
image = np.array(image_test)
image = np.transpose(image, (1, 0, 2))
# Preprocess the image to match the input shape of the model
image = np.expand_dims(image, axis=0)  # Add a batch dimension
image = image / 255.0  # Normalize the pixel values
prediction = model.predict(image)
# Print the predicted class label and probability
class_label = np.argmax(prediction)
class_prob = prediction[0][class_label]
if class_prob < 0.5: 
    y_output = 0 
else:
    y_output = 1
print(f"Class: {y_output}")

