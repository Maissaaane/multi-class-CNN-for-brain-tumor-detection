# Project: Brain Tumor Detection from MRI
This project uses a Convolutional Neural Network (CNN) model to classify brain MRI images, detecting the presence or absence of tumors. The model has been trained on a dataset of MRI images and is capable of making predictions on new images.

# Project Structure
Projet.ipynb: This notebook contains the main code for data preparation, as well as the building, training, and evaluation of the CNN model. It uses the TensorFlow and Keras libraries.

Test_projet.ipynb: This notebook demonstrates how to load the pre-trained model (tumor.keras) and use it to make predictions on a single image.

tumor.keras: The pre-trained Keras model, saved after training. It is ready for inference.

archive/: This folder should contain the MRI image dataset, structured into two subfolders: no (for images without tumors) and yes (for images with tumors).

# Dependencies
This project requires the following Python libraries:

tensorflow

keras

numpy

matplotlib

scikit-learn

You can install them using pip:

pip install tensorflow keras numpy matplotlib scikit-learn

# How to Use
Clone the Repository: Clone this repository to your local machine.

Organize the Data: Place your MRI image dataset in the archive/ folder, with no and yes subfolders.

Train the Model: Open and run the Projet.ipynb notebook to train the model. The model's parameters and architecture are defined there.

Test the Model: The Test_projet.ipynb notebook provides a simple example of using the saved model (tumor.keras) to make a prediction on a specific image (15no.jpg in the example).

# Model Summary
The core of this project is a Sequential Convolutional Neural Network (CNN) built with the Keras API. The architecture is designed to efficiently extract and learn features from the spatial patterns present in the MRI images.

The model consists of the following key components:

Convolutional Layers (Conv2D): Multiple convolutional layers are used to apply filters to the input images. These layers are responsible for detecting low-level features such as edges, textures, and shapes, which are crucial for identifying potential tumors.

Pooling Layers (MaxPooling2D): Max pooling is applied after the convolutional layers to reduce the dimensionality of the feature maps. This helps to make the model more computationally efficient and robust to variations in the position of the features.

Normalization and Dropout: Batch Normalization layers are included to stabilize and accelerate the training process. Dropout layers are also strategically placed to prevent overfitting by randomly deactivating a fraction of neurons during training, forcing the network to learn more robust features.

Dense Layers (Dense): After the feature extraction is complete, the flattened output is fed into one or more fully connected dense layers. These layers are responsible for performing the final classification based on the features learned by the convolutional layers.

Output Layer: The final dense layer uses a sigmoid activation function to output a single value between 0 and 1, representing the probability of a tumor being present. A value above 0.5 is typically interpreted as a positive tumor prediction.

The combined architecture allows the model to learn a hierarchical representation of the images, progressing from simple features to more complex patterns, which is ideal for this type de tâche de classification d'images.

# Author
Maïssane Frikh - Passionate about Deep Learning and its applications in the healthcare field.
