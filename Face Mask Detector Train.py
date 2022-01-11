# %%
# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import numpy as np
import os
import matplotlib.pyplot as plt
from imutils import paths


# %%
# path to the dataset, saved model and the plot of the training loss and accuracy
datasetPath = r'D:\ComputerVision\dataset'
savedModelPath = r'D:\ComputerVision\mobilenetv2_facemask.model'
savedPlotPath = r'D:\ComputerVision\plot_facemask.png'


# %% [markdown]
# ## Data Preprocessing

# %%
# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
imagePaths = list(paths.list_images(datasetPath))

data = []
labels = []

# loop over the image paths
for i in imagePaths:
    # extract the class label from the filename
    label = i.split(os.path.sep)[-2]

    # load the input image (224x224) and preprocess it
    # Pre-processing steps include resizing to 224×224 pixels
    image = load_img(i, target_size=(224, 224))
    # conversion to array format
    image = img_to_array(image)
    # and scaling the pixel intensities in the input image to the range [-1, 1] (via the preprocess_input convenience function)
    image = preprocess_input(image)

    # update the data and labels lists, respectively
    labels.append(label)
    data.append(image)


# %%
# convert the data and labels to NumPy arrays
data = np.array(data, dtype='float32')
labels = np.array(labels)


# %%
data.shape


# %%
np.unique(labels)


# %%
# perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)


# %%
# each element of our labels array consists of an array in which only one index is “hot”
labels


# %%
# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
train_X, test_X, train_Y, test_Y = train_test_split(
    data, labels, test_size=0.20, stratify=labels, random_state=42)


# %%
train_X.shape


# %%
train_Y.shape


# %%
test_X.shape


# %%
test_Y.shape


# %% [markdown]
# ## Data Training

# %%
# During training, we’ll be applying on-the-fly mutations to our images in an effort to improve generalization.
# This is known as data augmentation, where the random rotation, zoom, shear, shift, and flip parameters are established.
# We’ll use the aug object at training time.

# construct the training image generator for data augmentation
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest')


# %%
# Prepare MobileNetV2 for fine-tuning

# Step 1: Load MobileNet with pre-trained ImageNet weights, leaving off head of network

# load the MobileNetV2 network, ensuring the head FC layer sets are
# left off
baseModel = MobileNetV2(weights="imagenet", include_top=False,
                        input_tensor=Input(shape=(224, 224, 3)))

baseModel.summary()


# %%
# Step 2: Construct a new FC head, and append it to the base in place of the old head

# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name='Flatten')(headModel)
headModel = Dense(128, activation='relu')(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation='softmax')(headModel)

# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

model.summary()


# %%
# Step 3: Freeze the base layers of the network.
# The weights of these base layers will not be updated during the process of backpropagation, whereas the head layer weights will be tuned.

# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
    layer.trainable = False

model.summary()

# %%
# initialize the initial learning rate, number of epochs to train for,
# and batch size
INIT_LR = 1e-4
EPOCHS = 20
BS = 32


# %%
# Compile our model with the Adam optimizer, a learning rate decay schedule, and binary cross-entropy.

# compile our model
opt = Adam(learning_rate=INIT_LR, decay=INIT_LR/EPOCHS)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])


# %%
# Face mask training is launched. Our data augmentation object (aug) will be providing batches of mutated image data.

# train the head of the network
H = model.fit(
    aug.flow(train_X, train_Y, batch_size=BS),
    steps_per_epoch=len(train_X)//BS,
    validation_data=(test_X, test_Y),
    validation_steps=len(test_X)//BS,
    epochs=EPOCHS
)


# %%
# serialize the model to disk
model.save(savedModelPath, save_format="h5")

# %% [markdown]
# ## Model Evaluating

# %%
# Evaluate the resulting model on the test set

# make predictions on the testing set
predIdxs = model.predict(test_X, batch_size=BS)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

# show a nicely formatted classification report
print(classification_report(test_Y.argmax(
    axis=1), predIdxs, target_names=lb.classes_))


# %%
# plot the training loss and accuracy

N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="training_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="validation_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="training_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="validation_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(savedPlotPath)
