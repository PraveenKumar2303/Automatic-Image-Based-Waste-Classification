from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Sequential
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from keras.preprocessing import image



train_dir = r"C:\Users\PRAVEEN KUMAR C\OneDrive\Documents\IMARTICUS Class\Capstone Project 2\CNN (Bio & Non_Bio wastes)\Bio & Non_Bio Wastes\Train" # Directory containing the training data
test_dir = r"C:\Users\PRAVEEN KUMAR C\OneDrive\Documents\IMARTICUS Class\Capstone Project 2\CNN (Bio & Non_Bio wastes)\Bio & Non_Bio Wastes\Val"  # Directory containing the validation data

train_datagen = ImageDataGenerator(
    width_shift_range = 0.1,        # Randomly shift the width of images by up to 10%
    height_shift_range = 0.1,       # Randomly shift the height of images by up to 10%
    horizontal_flip = True,         # Flip images horizontally at random
    rescale = 1/255,                # Rescale pixel values to be between 0 and 1
    validation_split = 0.2,         # Set aside 20% of the data for validation
)

validation_datagen = ImageDataGenerator(
    rescale = 1/255,                # Rescale pixel values to be between 0 and 1
    validation_split = 0.2          # Set aside 20% of the data for validation
)

train_generator = train_datagen.flow_from_directory(
    directory = train_dir,           # Directory containing the training data
    target_size = (128, 128),        # Resizes all images to 128x128 pixels
    batch_size = 64,                 # Number of images per batch
    class_mode = "categorical",      # Classifies the images into 2 categories
    subset = "training"              # Uses the training subset of the data
)

validation_generator = validation_datagen.flow_from_directory(
    directory = test_dir,            # Directory containing the validation data
    target_size = (128, 128),        # Resizes all images to 128x128 pixels
    batch_size = 64,                 # Number of images per batch
    class_mode = "categorical",      # Classifies the images into 2 categories
    subset = "validation"            # Uses the validation subset of the data
)

import matplotlib.pyplot as plt

# Get a batch of images and labels from the generator
batch = next(train_generator)

# Display the images in a 4x4 grid with their corresponding labels
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    ax.imshow(batch[0][i]) # display the image at index i of the batch on the current axis
    label_idx = batch[1][i].argmax() # get the index of the label for the image at index i of the batch
    label_map = {v:k for k,v in train_generator.class_indices.items()} # create a dictionary mapping the label index to the corresponding label name
    ax.set_title(label_map[label_idx]) # set the title of the current axis to the label name corresponding to the label index
    ax.axis('off') # turn off the axis ticks and labels

plt.show() # display the figure

from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Sequential
import tensorflow as tf
# Define the model architecture
model = Sequential()
# Add a convolutional layer with 32 filters, 3x3 kernel size, and relu activation function
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(128,128,3)))
# Add a batch normalization layer
model.add(BatchNormalization())
# Add a second convolutional layer with 64 filters, 3x3 kernel size, and relu activation function
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
# Add a second batch normalization layer
model.add(BatchNormalization())
# Add a max pooling layer with 2x2 pool size
model.add(MaxPooling2D(pool_size=(2, 2)))
# Add a dropout layer with 0.25 dropout rate
model.add(Dropout(0.25))

# Flatten the output of the convolutional layers
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='sigmoid'))

# Compile the model with categorical cross-entropy loss, Adam optimizer, and accuracy metric
model.compile(loss="categorical_crossentropy", optimizer= tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])

# Model Summary
model.summary()

from tensorflow.keras.callbacks import ModelCheckpoint

# Define the callback to save the model's weights
# The filepath specifies where the weights should be saved
# The monitor argument specifies the metric to monitor for saving the best weights
# The save_best_only argument ensures that only the weights with the highest monitored metric value are saved
# The save_weights_only argument specifies to only save the weights (not the entire model)
# The mode argument specifies whether to maximize or minimize the monitored metric value (in this case, 'max' for accuracy)
checkpoint_callback = ModelCheckpoint(
    filepath='model.weights.h5',
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=True,
    mode='max'
)
# The callbacks argument specifies the callbacks to use during training (in this case, just the ModelCheckpoint callback)
history = model.fit(
    train_generator,
    batch_size=64,
    epochs=15,
    validation_data=validation_generator,
    callbacks=[checkpoint_callback]
)

# Generate generalization metrics
validation_loss, validation_acc = model.evaluate(validation_generator)
print('Validation loss:', validation_loss, '\t Validation accuracy:', validation_acc)

import seaborn as sns
from sklearn.metrics import confusion_matrix

# Get the true labels and predicted labels for the validation set
validation_labels = validation_generator.classes
validation_pred_probs = model.predict(validation_generator)
validation_pred_labels = np.argmax(validation_pred_probs, axis=1)

# Compute the confusion matrix
confusion_mtx = confusion_matrix(validation_labels, validation_pred_labels)
class_names = list(train_generator.class_indices.keys())
#sns.set()
sns.heatmap(confusion_mtx, annot=True, fmt='d', cmap='YlGnBu',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

model.save("model.h5")

# Load an image to predict
img_path = "/content/drive/MyDrive/CNN (Bio & Non_Bio wastes)/Val/Non_Biodegradable/TEST_NBIODEG_ORI_3614.jpg"
img = image.load_img(img_path, target_size=(128, 128))  # Load image and resize it to (48, 48) and convert it to grayscale
img_array = image.img_to_array(img)  # Convert the image to a numpy array
img_array = np.expand_dims(img_array, axis=0)  # Add an extra dimension to the array to make it suitable for model input
img_array /= 255.  # Normalize the pixel values between 0 and 1

# Make a prediction
prediction = model.predict(img_array)  # Predict the emotion label for the image

# Get the predicted label
label_map = {v:k for k,v in train_generator.class_indices.items()}  # Map the class indices to the corresponding emotion labels
predicted_label = label_map[np.argmax(prediction)]  # Get the emotion label with the highest predicted probability
print(" Waste predicted of this image is", predicted_label)  # Print the predicted emotion label

img

