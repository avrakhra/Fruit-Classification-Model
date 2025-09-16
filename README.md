# Fresh vs Spoiled Fruit Classification Model

## Project Overview
This project uses a Convolutional Neural Network (CNN) to classify images of fruits as either fresh or rotten. The model is built in Python using TensorFlow/Keras, trained on a dataset of fruit images.

## Dataset
- Source: [Fruits Quality (Fresh VS Rotten)](https://www.kaggle.com/datasets/nourabdoun/fruits-quality-fresh-vs-rotten)
- Training set: 287 images
- Validation set: 48 images
- Test set: 24 images

## Features and Model Architecture

**Data Augmentation (applied to training data only):**
- Rotation
- Width/Height Shift
- Horizontal Flip
- Shear 
- Zoom

**CNN Architecture:**
1. **Convolutional Layer + MaxPooling2D**
    - Detects basic patterns like edges and shapes.
2. **Second Convolutional Layer + MaxPooling2D**
    - Detects higher-level patterns.
3. **Flatten + Dense + Dropout**
    - Flatten: converts 2D feature maps to 1D vector.
    - Dense: fully connected layer with 128 neurons. 
    - Dropout: 50% of neurons are randomly ignored to prevent overfitting.
4. **Output Layer**
    - Produces probability for binary classification (0 = fresh, 1 = spoiled)

**Results**
- Training accuracy: ~83%
- Validation accuracy: ~70%
- Test accuracy: ~73%
- Observations: Model learns patterns of fresh vs spoiled fruit, but is not perfect due to small dataset size.

**How to Run**
1. Clone this repository:
``` git clone <repo-url>
```
2. Install dependencies
``` pip install tensorflow numpy
```
3. Run the Python script:
``` python main.py
```
4. Predictions will be printed with confidence scores.


**Notes**
- Model saves automatically as "fruit_classification_model.keras"
- Rerunning the script will load the existing model and continue training from there

**Main Takeaways**
- The CNN model achieved ~73% test accuracy, showing it can reasonably distinguish fresh vs. spoiled fruit.
- Increasing the dataset size would likely improve model generalization.
- Data augmentation helped the model learn better from a small dataset.
- Some misclassifications show that subtle differences between fresh and spoiled fruits are challenging for the model. 

