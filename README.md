# MNIST Digit Recognizer

## Model Architecture

The model architecture used in this project is a convolutional neural network (CNN) consisting of two convolutional layers followed by max-pooling layers and two fully connected layers. Here is a summary of the architecture:

1. Convolutional Layer 1:
   - Input: 1 channel (grayscale image)
   - Output: 32 channels
   - Kernel size: 3x3
   - Padding: 1
   - Activation function: ReLU

2. Max Pooling Layer 1:
   - Kernel size: 2x2

3. Convolutional Layer 2:
   - Input: 32 channels
   - Output: 64 channels
   - Kernel size: 3x3
   - Padding: 1
   - Activation function: ReLU

4. Max Pooling Layer 2:
   - Kernel size: 2x2

5. Fully Connected Layer 1:
   - Input: 64 * 7 * 7 (output from previous layers)
   - Output: 128
   - Activation function: ReLU

6. Fully Connected Layer 2:
   - Input: 128
   - Output: 10 (number of classes)
   - Activation function: LogSoftmax

## Training Process

The model was trained using the MNIST dataset with data augmentation techniques such as random rotation and affine transformations. The training process involved optimizing the model using the Adam optimizer with a learning rate of 0.001 and a cross-entropy loss function. The training was performed for 20 epochs.

## Results

After training, the model achieved an accuracy of approximately 98% on the test set, demonstrating its ability to accurately classify handwritten digits.

## Visualization Interface

To use the visualization interface:
1. Run the `app.py` script using Streamlit.: `streamlit run app.py`
2. Choose the input method (`Canvas` or `Upload from folder`) from the sidebar.
3. If using the canvas, draw a digit and let the model predict it. If uploading from a folder, select an image from the dropdown list.
4. The predicted digit will be displayed along with the drawn/uploaded image.

## Here the link for the live website:

https://mnistneuralnetwork-jayachandranpm.streamlit.app/


---

**Developed by Jayachandran P M**
