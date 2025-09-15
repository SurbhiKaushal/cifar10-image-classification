# CIFAR-10 Image Classification

## Dataset
For this project, I used the CIFAR-10 dataset. It contains 60,000 color images in 10 different categories such as airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. The dataset is widely used for image classification tasks and is a great choice to train and test a simple convolutional neural network.

## Preprocessing
I normalized the images by scaling the pixel values from 0–255 to 0–1, which helps the model learn better. The labels were one-hot encoded since the model needs to classify the images into one of 10 categories.

## Workflow
1. Loaded the dataset using TensorFlow.
2. Explored the dataset by printing the shapes and visualizing sample images.
3. Preprocessed the images by normalization and encoding the labels.
4. Built a convolutional neural network (CNN) with two convolutional layers, max-pooling layers, and dense layers.
5. Trained the model for 10 epochs using the Adam optimizer and categorical crossentropy loss.
6. Evaluated the model's accuracy on the test set.
7. Plotted the training and validation accuracy to understand the learning process.

## Results
After training the model, it achieved a test accuracy of around **XX%** (replace this with your actual result after running the code). The accuracy graph shows how the model improved over the course of training, which confirms that the model is learning effectively.

## How to Run
1. Make sure you have Python installed along with TensorFlow and Matplotlib. You can install them using:
