# This is the learning process for TensorFlow's Keras, entirely based on the official TensorFlow website content.
#  After learning, I will use the inner power system from the Ni Shui Han mobile game as my experiment subject.
# Firstly, TensorFlow is a machine learning platform, and we are using Keras, which is a neural network model. 
# On the official website, it is introduced using clothing and shoe classification. Let's take a look.
# TensorFlow and tf.keras: this tf.keras is an API for learning.
import tensorflow as tf
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
print(tf.__version__)
# The above is the initial setup, here the version is displayed as 2.18.0
# Then, it uses a dataset for classifying clothing, shoes, bags, etc. Fashion MNIST. 
# Each clothing item and shoe is a 28x28 grayscale image, 10 categories, 70,000 images.
# It is expected to use 60,000 images for training and 10,000 images to evaluate the performance.
# OK, import the dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
# The following code is interesting, why does it automatically split into two sets? 
# Actually, this dataset was already split into 60,000 and 10,000 training and testing sets when created. 
# fashion_mnist.load_data() simply loads the training set and places them into two different sets.
# (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# Let's try something else, what if we only put it into one set? What happens? I'll try the code below. 
# It seems it can run, so let's see what the structure of this dataset is.

# (images, labels) = fashion_mnist.load_data()
# print(images.shape)
# print(labels.shape)

# Unfortunately! The result is:
# "Traceback (most recent call last):
#   File "c:\Users\64171\Desktop\TensorFlow入门.py", line 17, in <module>
#     print(images.shape)
#     ^^^^^^^^^^^^
# AttributeError: 'tuple' object has no attribute 'shape'”

# Wow, so we understand that actually, fashion_mnist.load_data returns a tuple, 
# something like "((train_images, train_labels), (test_images, test_labels))". We can see this is a tuple containing two tuples, 
# so if we directly assign it to (images, labels), then images get a tuple (train_images, train_labels), 
# and labels get a tuple (test_images, test_labels), which causes the error. So, we still need to assign it as before:
# (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Alright, let's comment out the above and continue following the introductory guide.

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
print(train_images.shape) #(60000, 28, 28)
print(train_labels.shape) #(60000,)
print(test_images.shape) #(10000, 28, 28)
print(test_labels.shape) #(10000,)

# Very good, we see the above results. The image has the first number indicating how many data points, 
# the second and third numbers are the structure of the image, 28*28 pixels. Let's further explore what this dataset really is.
print(train_labels[1:10])  # See the first 10 labels
print(train_images[0])  # See the first image
print(train_images[0].shape) # See the shape of the first array
print(np.unique(train_labels))  # See what values exist

# The results are as follows: too long, I'll use green comments instead.
# First, let's look at the first ten labels, oh, they are the categories each of the first ten images belongs to.
# [0 0 3 0 2 7 2 5 5]

# Secondly, let's see what the first data exactly is, wow, it's a 2D array. Through print(train_images[0].shape), 
# we see it's a 28*28 array, 0 represents white, 255 represents black. This 28*28 pixel matrix forms an image, this is our
# image!

# [[  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
#     0   0   0   0   0   0   0   0   0   0]
#  [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
#     0   0   0   0   0   0   0   0   0   0]
#  [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
#     0   0   0   0   0   0   0   0   0   0]
#  ...
#  [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
#     0   0   0   0   0   0   0   0   0   0]]

# Thirdly, let's see what categories there are, there are ten categories in total
# [0 1 2 3 4 5 6 7 8 9]
# Specifically, they represent 'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'. 
# Very good! Everything is clear.

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# OK, let's take a closer look at the first image and plot it!
plt.figure() # Create a plot window. You can add parameters like figsize=(n, n) to set the window size to n*n inches
plt.imshow(train_images[0]) # Matplotlib will map this 2D array to an image by default.
plt.colorbar() # Add a color bar next to the image. 
# Note that the plotted image is not black and white but colored because Matplotlib uses a colormap called viridis by default. 
# We can add cmap='gray' in the above code to make the image grayscale.
# I will continue to follow the official guide's code
plt.grid(False) # By default, grid lines are shown. Turning them off makes the image clearer, currently turned off
plt.show() # Display the image

# Now, for training, we need to scale the pixel values to between 0-1, that is, divide by 255
train_images = train_images / 255.0
test_images = test_images / 255.0

plt.figure(figsize=(10,10)) # Here, figsize is used to set the window size, possibly because there are many images.
for i in range(25):  # Generate numbers from 0-24, then loop
    plt.subplot(5,5,i+1)  # In this window, draw subplots. The subplot grid is 5*5, which can fit 25 images! 
    # i+1 indicates the current subplot number, so the first one is 1. The numbering is necessary to place the following images in order. 
    # i+1 is required, otherwise it errors!
    plt.xticks([]) # Hide x-axis ticks
    plt.yticks([]) # Hide y-axis ticks
    plt.grid(False) # Same as before, turn off grid lines
    plt.imshow(train_images[i], cmap=plt.cm.binary) # Plot the object as train_images[i]. cmap=plt.cm.binary is grayscale mode. 
    # Similar to 'gray', but the brightness may differ a bit, both are fine.
    plt.xlabel(class_names[train_labels[i]]) # Add a label to the subplot's x-axis, taking the corresponding label of the i-th image. 
    # train_labels[i] will get an integer from 0-9, then the class_names list gets the corresponding name using this number
plt.show()

# Next, we start building the model
model = tf.keras.Sequential([  # Use Keras' Sequential API to build a sequential model. Input data will pass through layers one by one.
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # First layer, convert the 28*28 2D array into a 1D array of 784 pixels, just reshaping.
    tf.keras.layers.Dense(128, activation='relu'),  # Fully connected layer with 128 neurons.
    #  Each neuron receives all the inputs from the previous layer. Uses ReLU activation function, meaning if input <0, take 0; if input >=0, take the input
    # However, my previous input pixels are between 0-1, after processing by the neurons, negative values may appear.
    tf.keras.layers.Dense(10)  # Output layer with 10 neurons, corresponding to 10 categories. 
    # Each neuron outputs a score for the corresponding category, which is unnormalized, so values may be large or small, any real number.
])
# Additionally, regarding the weight matrix, what is its shape? 784*128, i.e., 784 rows and 128 columns. 
# For example, the first row represents 128 weights for the first pixel to 128 neurons. Each column has 784 weights for the first neuron, and so on, forming a 784*128 matrix.

# As for activation, it's a bit confusing. Let me see carefully. For a single neuron, it's like z = w1 * x1 + w2 * x2 + ... + wn * xn + b. 
# Using activation, it becomes y = activation(z) = activation(w1 * x1 + w2 * x2 + ... + wn * xn + b).
#  The bias term is to allow the neuron to have a baseline output even when inputs are 0. Without activation, z and x1, x2,... are linearly related. 
# Activation makes it non-linear; ReLU is the simplest and fastest, it turns negatives to non-negatives.

# Compile the model
model.compile(optimizer='adam',  # model.compile configures the model's training method, specifying the following three items. 
# First, optimizer is 'adam', which is to minimize prediction error. The specific operation of 'adam' is complex and not understood.
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              # Loss function to measure the difference between model predictions and true labels. Using SparseCategoricalCrossentropy loss function, from_logits=True
              # This tells the function that the output is unnormalized; it should apply softmax to calculate. 
              # Note that SparseCategoricalCrossentropy is suitable for integer-encoded labels; one-hot encoded vectors are not suitable
              metrics=['accuracy']) # Evaluation metric, using accuracy, which is correct count / total sample count

# Start training the data
model.fit(train_images, train_labels, epochs=10) # epochs refers to the number of times the model learns the dataset. 
# Here, the model will learn 10 times. Other available parameters like batch_size can set the amount of data used per training.

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2) # Evaluate model performance. 
# The verbose parameter can be 0, 1, or 2. 0 means no output, 1 means progress bar and evaluation results, 2 means only evaluation results

print('\nTest accuracy:', test_acc) #313/313 - 0s - 821us/step - accuracy: 0.8866 - loss: 0.3357 Test accuracy: 0.8866000175476074

# Next, we can use the trained model for predictions
# Create a probability model, the first layer is the original model, the second layer applies softmax to normalize, 
# meaning the unnormalized outputs are converted to normalized probabilities.
probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images) # Test images, predictions become a collection of results
predictions[0] # Look at the first result
np.argmax(predictions[0]) # Note, the result above should be a 1D array of ten numbers, here we find the one with the highest value, which is the most likely one.

# Plot bar chart to show prediction probabilities for each category
def plot_image(i, predictions_array, true_label, img): # i is the index, predictions_array is the prediction vector
    true_label, img = true_label[i], img[i]
    plt.grid(False) # Turn off grid lines, discussed earlier
    plt.xticks([])  # Hide x-axis ticks, discussed earlier
    plt.yticks([])  # Hide y-axis ticks, discussed earlier

    plt.imshow(img, cmap=plt.cm.binary) # Grayscale image, discussed earlier

    predicted_label = np.argmax(predictions_array) # Get the index with the highest prediction probability
    if predicted_label == true_label: # Use blue and red to indicate if the prediction is correct
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],  # Model's predicted label
                                        100*np.max(predictions_array), # Display the corresponding prediction probability
                                        class_names[true_label]), # Get the true label's class name to compare with the model's prediction
               color=color) # Set the label's color

def plot_value_array(i, predictions_array, true_label): # Similarly, i is the index, predictions_array is the prediction probability distribution vector.
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10)) # Indicate categories
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777") # Default gray color. 'thisplot' is used to control each bar's color later
    plt.ylim([0, 1]) # y-axis range 0-1, because we used softmax to normalize

    predicted_label = np.argmax(predictions_array) 

    thisplot[predicted_label].set_color('red')  # Highlight the predicted label's bar in red
    thisplot[true_label].set_color('blue')  # Highlight the true label's bar in blue

num_rows = 5  # Same as before to show training set: create five rows and three columns, plotting fifteen images
num_cols = 3
num_images = num_rows*num_cols

plt.figure(figsize=(2*2*num_cols, 2*num_rows)) # Set the canvas size. There are two plots per column - the image and its probability bar chart

for i in range(num_images): # Similarly, loop to plot the clothing image and its probability bar chart
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout() # Automatically adjust whitespace for a compact layout
plt.show()

img = test_images[1] # Try the first image

img = (np.expand_dims(img,0)) # Expand dimensions to convert the 2D array into a 3D array

predictions_single = probability_model.predict(img)

print(predictions_single)

plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45) # Rotate labels by 45 degrees for easier reading
plt.show()

np.argmax(predictions_single[0])

# Now, let's do my own part. In the Nixuan Cold mobile game, there are many inner powers, 
# but they belong to one of the five categories: Metal, Wood, Water, Fire, Earth. Some inner powers combine two elements.
#  I want to try whether through the appearance, the machine can recognize what element the inner power is.
import os
from PIL import Image
image_library = "C:/Users/64171/Desktop/Elements" # This is my directory
# Below, I put different inner powers into different subfolders
categories = ['金','木','水','火','土']

# First, create empty lists to gradually add inner power data to the dataset
data = []
labels = []

for label, category in enumerate(categories): # Now, we start traversing each folder. 
    # Note, the enumerate function automatically assigns an index to the result, that is, as each folder is traversed, it automatically gets a label.
    # So, each image will be tagged with the corresponding folder's label
    file_path = os.path.join(image_library, category) # Merge the path, build each category's subfolder path.
    for image in os.listdir(file_path):
       image_path = os.path.join(file_path, image) # Further merge to create each image's file path.
       image = Image.open(image_path).convert('L')   # Open the image and convert it to grayscale
       image_array = np.array(image) # Similarly, convert it to an array
       data.append(image_array) # Add the 2D array to the dataset
       labels.append(label)

data = np.array(data)/255 # Same as before, scale pixel values to 0-1
labels = np.array(labels) # Convert this list to a 1D array
print(data.shape)
print(labels.shape)

# (34, 138, 138)
# (34,)
# The above is the result, still successful, 34 images. Next, let's build the model.

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(138, 138)),  # Input images are 138*138, so change to 138
    tf.keras.layers.Dense(128, activation='relu'),    # Same as before
    tf.keras.layers.Dense(5)    # We only have five categories: Metal, Wood, Water, Fire, Earth, so change to 5
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping( # This function is easy to understand, the parameters are standard English terms.
    monitor='val_loss',  # Monitor the loss metric
    patience=5,          # If the validation loss does not improve for 5 consecutive epochs, stop training
    restore_best_weights=True  # Restore the weights from the epoch with the best validation performance
)
model.fit(data, labels, epochs=100, callbacks=[early_stopping])

# Here, I found a problem. I initially set to train ten epochs, but the results showed:
# “Epoch 1/10
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 24ms/step - accuracy: 0.1801 - loss: 2.8419 
# Epoch 2/10
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step - accuracy: 0.2102 - loss: 14.5704
# Epoch 3/10
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step - accuracy: 0.3303 - loss: 6.6725
# Epoch 4/10
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step - accuracy: 0.1697 - loss: 9.9408 
# Epoch 5/10
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step - accuracy: 0.1501 - loss: 5.7031
# Epoch 6/10
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step - accuracy: 0.2598 - loss: 2.9296
# Epoch 7/10
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step - accuracy: 0.1998 - loss: 3.0455
# Epoch 8/10
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step - accuracy: 0.2102 - loss: 3.4229
# Epoch 9/10
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step - accuracy: 0.2102 - loss: 1.9663
# Epoch 10/10
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step - accuracy: 0.3094 - loss: 2.3802”
# You can see the accuracy keeps fluctuating, so I increased the number of training epochs, as follows:
# “Epoch 1/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 29ms/step - accuracy: 0.1201 - loss: 3.1592 
# Epoch 2/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step - accuracy: 0.3199 - loss: 22.0484
# Epoch 3/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step - accuracy: 0.3603 - loss: 17.8683
# Epoch 4/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step - accuracy: 0.1998 - loss: 12.2298
# Epoch 5/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step - accuracy: 0.3002 - loss: 3.1205
# Epoch 6/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step - accuracy: 0.2598 - loss: 3.9203
# Epoch 7/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step - accuracy: 0.1893 - loss: 5.3345
# Epoch 8/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step - accuracy: 0.2702 - loss: 4.7120
# Epoch 9/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step - accuracy: 0.1801 - loss: 2.6470
# Epoch 10/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step - accuracy: 0.1998 - loss: 3.5455
# Epoch 11/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step - accuracy: 0.4504 - loss: 1.7801
# Epoch 12/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step - accuracy: 0.2402 - loss: 2.2252
# Epoch 13/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step - accuracy: 0.5404 - loss: 1.4328
# Epoch 14/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step - accuracy: 0.2898 - loss: 2.0863
# Epoch 15/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step - accuracy: 0.4295 - loss: 1.4313
# Epoch 16/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step - accuracy: 0.3002 - loss: 1.2737
# Epoch 17/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step - accuracy: 0.2598 - loss: 1.5952
# Epoch 18/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step - accuracy: 0.4099 - loss: 1.2606
# Epoch 19/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step - accuracy: 0.2898 - loss: 1.3698
# Epoch 20/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step - accuracy: 0.4203 - loss: 1.5257
# Epoch 21/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step - accuracy: 0.3499 - loss: 1.2937
# Epoch 22/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step - accuracy: 0.4203 - loss: 1.3803
# Epoch 23/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step - accuracy: 0.3799 - loss: 1.6801
# Epoch 24/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - accuracy: 0.5196 - loss: 1.1974
# Epoch 25/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step - accuracy: 0.3002 - loss: 1.5174
# Epoch 26/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step - accuracy: 0.5600 - loss: 0.9484
# Epoch 27/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step - accuracy: 0.3499 - loss: 1.2916
# Epoch 28/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step - accuracy: 0.1998 - loss: 1.6019
# Epoch 29/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step - accuracy: 0.6397 - loss: 1.0284
# Epoch 30/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step - accuracy: 0.6906 - loss: 0.9794
# Epoch 31/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step - accuracy: 0.4596 - loss: 1.1767
# Epoch 32/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step - accuracy: 0.6501 - loss: 0.8682
# Epoch 33/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step - accuracy: 0.6605 - loss: 0.8500
# Epoch 34/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step - accuracy: 0.6501 - loss: 0.9618
# Epoch 35/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step - accuracy: 0.7298 - loss: 0.9010
# Epoch 36/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step - accuracy: 0.5300 - loss: 0.9587
# Epoch 37/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step - accuracy: 0.5000 - loss: 1.0649
# Epoch 38/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step - accuracy: 0.7298 - loss: 0.7586
# Epoch 39/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step - accuracy: 0.5901 - loss: 1.0501
# Epoch 40/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - accuracy: 0.4099 - loss: 1.3673
# Epoch 41/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step - accuracy: 0.4700 - loss: 1.1719
# Epoch 42/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step - accuracy: 0.6097 - loss: 0.7892
# Epoch 43/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step - accuracy: 0.7702 - loss: 0.7647
# Epoch 44/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step - accuracy: 0.4896 - loss: 1.1093
# Epoch 45/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step - accuracy: 0.7598 - loss: 0.7785
# Epoch 46/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step - accuracy: 0.6998 - loss: 0.7460
# Epoch 47/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step - accuracy: 0.4596 - loss: 1.1263
# Epoch 48/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step - accuracy: 0.4203 - loss: 1.0210
# Epoch 49/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step - accuracy: 0.7506 - loss: 0.7264
# Epoch 50/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step - accuracy: 0.5600 - loss: 1.0391”
# You can see the accuracy keeps fluctuating, so I increased the number of training epochs, as follows:
# “Epoch 1/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 29ms/step - accuracy: 0.1201 - loss: 3.1592 
# Epoch 2/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step - accuracy: 0.3199 - loss: 22.0484
# [...]
# Epoch 50/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step - accuracy: 0.5600 - loss: 1.0391”
# The accuracy kept fluctuating, so I added an "EarlyStopping" mechanism

#? from tensorflow.keras.callbacks import EarlyStopping

#? early_stopping = EarlyStopping( # This function is easy to understand, the parameters are standard English terms.
#?     monitor='val_loss',  # Monitor the loss metric
#?     patience=5,          # If the validation loss does not improve for 5 consecutive epochs, stop training
#?     restore_best_weights=True  # Restore the weights from the epoch with the best validation performance
#? )
#? Certainly, I directly added the above content to the front.

# OK, awesome, I ran 100 epochs, and it directly got the accuracy to 100%!
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step - accuracy: 1.0000 - loss: 0.2569

# Next, we'll create our test set in exactly the same way

test_library = "C:/Users/64171/Desktop/ETEST"  
categories = ['金', '木', '水', '火', '土']  

test_data = []
test_labels = []

for label, category in enumerate(categories):
    file_path = os.path.join(test_library, category)  
    for image in os.listdir(file_path):
        image_path = os.path.join(file_path, image)  
        image = Image.open(image_path).convert('L')  
        image_array = np.array(image)  
        test_data.append(image_array)  
        test_labels.append(label)  

test_data = np.array(test_data)/255.0  
test_labels = np.array(test_labels)  

print(test_data.shape)  
print(test_labels.shape)  

# Then, we start building the probability model
label_dictionary = {label: category for label, category in enumerate(categories)}
print(label_dictionary)
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()]) # Exactly the same as before
predictions = probability_model.predict(test_data)
print(predictions[0]) 
np.argmax(predictions[0])
print(np.unique(test_labels))
# At this point, I realized that although I have mapped the numbers to categories, I don't know which category corresponds to which label, so I added this at the front:
# label_dictionary = {label: category for label, category in enumerate(categories)}

# Next, I will perform testing. I used another 13 test inner powers. The code is exactly the same as before!
class_names = ['金', '木', '水', '火', '土']

def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)  

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'  
    else:
        color = 'red'  

    plt.xlabel("{} {:2.0f}% ({})".format(
        class_names[predicted_label],  
        100 * np.max(predictions_array),  
        class_names[true_label] 
    ), color=color)  


def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(len(class_names)), class_names, rotation=45)  
    plt.yticks([])
    thisplot = plt.bar(range(len(class_names)), predictions_array, color="#777777")
    plt.ylim([0, 1])  

    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')  
    thisplot[true_label].set_color('blue')  


num_rows = 5
num_cols = 3
num_images = 13 # All content remains unchanged, only here it changes because I only have 13 inner powers for testing!

plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions[i], test_labels, test_data)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

# Alright, the final results are out, the performance is dismal, the accuracy is only about half. 
# However, I found that among the five categories, Metal and Earth have five and were all predicted correctly, 
# while Wood, Water, and Fire were almost all wrong. So I think, to some extent, maybe Metal and Earth share many similar features internally, 
# for example, Earth inner powers might represent mountains, while Metal inner powers represent blades and weapons! I'll post the final results online.

# To correct, I tried different numbers of training epochs and found that Earth inner powers are not that stable either. Metal inner powers are almost always correct!