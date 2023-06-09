# Implementation-of-Transfer-Learning
## Aim
To Implement Transfer Learning for CIFAR-10 dataset classification using VGG-19 architecture.
## Problem Statement and Dataset
To use the pre-trained VGG19 model on the ImageNet dataset and fine-tune it on the CIFAR-10 dataset. The goal is to achieve high accuracy on the CIFAR-10 dataset by leveraging the knowledge learned from the ImageNet dataset. The main challenge is to adapt the pre-trained model to the new dataset while avoiding overfitting and achieving high accuracy

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class. the datasetcontains 10 random images of:

airplane

automobile

bird

cat

deer

dog

frog

horse

ship

truck

## DESIGN STEPS
### STEP 1:
Import the required libraries and load the dataset

### STEP 2:
split the dataset for training and testing

### STEP 3:
Set the values of image from 0 t0 1.

### STEP 4:
using the VGG 19 as base model without changing the weights and remove the fully connected layer from VGG19

### STEP 5:
Add our own Fully connected layer to VGG19 base model, compile and fit it


## PROGRAM

~~~
import pandas as pd
import numpy as np
from keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG19
from keras import Sequential
from keras.layers import Flatten,Dense,BatchNormalization,Activation,Dropout
from tensorflow.keras import utils
from sklearn.metrics import classification_report,confusion_matrix

(x_train,y_train),(x_test,y_test)=cifar10.load_data()

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0


# One Hot Encoding Outputs
y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)

base_model=VGG19(
    include_top=False,
    weights='imagenet',
    input_shape=(32,32,3)
)

for layer in base_model.layers:
  layer.trainable = False


model=Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(800,activation=('relu')))
model.add(Dense(650,activation=('relu')))
model.add(Dropout(0.3))
model.add(Dense(500,activation=('relu')))
model.add(Dense(350,activation=('relu')))
model.add(Dense(200,activation=('relu')))
model.add(Dense(110,activation=('relu')))
model.add(Dropout(0.4))
model.add(Dense(98,activation=('relu')))
model.add(Dense(40,activation=('relu')))
model.add(Dense(10,activation=('softmax')))

# Create the model using transfer learning


model.summary()

from keras.optimizers import Adam
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

from tensorflow.keras.callbacks import ReduceLROnPlateau
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.00001)


model.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_test, y_test), callbacks=[learning_rate_reduction])

metrics = pd.DataFrame(model.history.history)

metrics[['loss','val_loss']].plot()

metrics[['accuracy','val_accuracy']].plot()

x_test_predictions = np.argmax(model.predict(x_test), axis=1)

print(confusion_matrix(y_test,x_test_predictions))

print(classification_report(y_test,x_test_predictions))

~~~


## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot
![p](loss.png)

![p](accuracy.png)

### Classification Report

![p](report.png)
### Confusion Matrix

![p](confusion.png)

## Conculsion:
We got an Accuracy of 60% with this model.There could be several reasons for not achieving higher accuracy. Here are a few possible explanations:

Dataset compatibility:
VGG19 was originally designed and trained on the ImageNet dataset, which consists of high-resolution images. In contrast, the CIFAR10 dataset contains low-resolution images (32x32 pixels). The difference in image sizes and content can affect the transferability of the learned features. Pretrained models like VGG19 might not be the most suitable choice for CIFAR10 due to this disparity in data characteristics.

Inadequate training data:
If the CIFAR10 dataset is relatively small, it may not provide enough diverse examples for the model to learn robust representations. Deep learning models, such as VGG19, typically require large amounts of data to generalize well. In such cases, you could consider exploring other architectures that are specifically designed for smaller datasets, or you might want to look into techniques like data augmentation or transfer learning from models pretrained on similar datasets.

Model capacity:
VGG19 is a deep and computationally expensive model with a large number of parameters. If you are limited by computational resources or working with a smaller dataset, the model's capacity might be excessive for the task at hand. In such cases, using a smaller model architecture or exploring other lightweight architectures like MobileNet or SqueezeNet could be more suitable and provide better accuracy.
## RESULT
Thus the implementation of Transfer learning of VGG19 for CIFAR10 dataset is successful.

