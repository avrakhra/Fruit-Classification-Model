import tensorflow
import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator


train_path = "dataset/train/"
val_path = "dataset/valid/"
test_path = "dataset/test/"

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=.2,
    height_shift_range=.2,
    horizontal_flip=True,
    shear_range=.2,
    zoom_range=.2
)

val_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_path, 
    target_size = (150, 150), 
    batch_size=8, 
    class_mode='binary',
    shuffle=True
)

val_data = val_datagen.flow_from_directory(
    val_path, target_size = (150, 150),
    batch_size=8,
    class_mode='binary'
)

test_data = test_datagen.flow_from_directory(
    test_path,
    target_size = (150, 150),
    batch_size=8, 
    class_mode='binary',
    shuffle=False 
)

# Build CNN model
model_file = "fruit_classification_model.keras"

if os.path.exists(model_file):
    model = load_model(model_file)
    print("Loaded existing model.")
else: 
    model = Sequential()
    # First convolutional layer: detects basic patterns (edges, colors)
    model.add(Conv2D(32, (3, 3), activation=("relu"), input_shape=(150, 150, 3))) # 32 filters each with 3x3 pixels this produces 32 feature maps showing different patterns detected
    model.add(MaxPooling2D(2, 2)) # Downsample feature maps

    # Second convolutional layer: detects higher-level patterns (shapes, textures)
    model.add(Conv2D(64, (3, 3), activation=("relu")))
    model.add(MaxPooling2D(2, 2))

    # Flatten and dense layers for classification
    model.add(Flatten()) 
    model.add(Dense(128, activation=("relu")))
    model.add(Dropout(.5)) # Prevent overfitting

    # Output layer: single neuron for binary classifcation
    model.add(Dense(1, activation='sigmoid')) 
    print("Created new model.")

# Compile model: tells the model how to learn and track performance
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
) 

# Train model
model_data = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20 # number of times model sees the training data
)

model.save("fruit_classification_model.keras")

# Evaluate model on test data
test_loss, test_acc = model.evaluate(test_data)
print("Test accuracy: ", test_acc)

# Make predictions on one batch of test images
predictions = model.predict(test_data)
labels = test_data.classes

print(train_data.class_indices)

if train_data.class_indices['fresh'] == 0:
    label_0 = "Fresh" 
    label_1 = "Spoiled"
else: 
    label_0 = "Spoiled" 
    label_1 = "Fresh"

# Loop through the first 5 images in the batch
for i in range(5):

    # Determine predicted label
    if predictions[i][0] >= 0.5:
        pred_label = label_1
    else: 
        pred_label = label_0
    
    # Determine actual label
    if labels[i] == 0:
        actual_label = label_0
    else: 
        actual_label = label_1

    # Get confidence score
    confidence = predictions[i][0]

    print("Image number: ", i+1)
    print("Model prediction: ", pred_label)
    print("Prediction confidence: ", confidence)
    print("Actual label:", actual_label)
    print()

    







