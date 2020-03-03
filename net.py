from tensorflow.keras import Sequential, datasets
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt


# Loading data
data = datasets.fashion_mnist

# Splitting data into training and testing data
(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Reshaping arrays to 4-dims
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

# Scaling data / normalizing
train_images = train_images / 255.0
test_images = test_images / 255.0

# Data info
print('Shape of training set: ', train_images.shape)
print('Shape of testing set: ', test_images.shape)

# Creating the Model
model = Sequential([
    Conv2D(28, kernel_size=(3, 3), input_shape=input_shape),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.25),
    Dense(10, activation='softmax')
])

# Training the Model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10)

# Testing the Model
scores = model.evaluate(test_images, test_labels)
print('\nTest loss:', scores[0])
print('\nTest accuracy:', scores[1])

# Making predictions
predictions = model.predict(test_images)

plt.figure(figsize=(5, 5))
for i in range(3):
    plt.grid(False)
    plt.imshow(test_images[i].reshape(28, 28), cmap=plt.cm.binary)
    plt.xlabel(class_names[test_labels[i]])
    plt.title(class_names[np.argmax(predictions[i])])
    plt.show()
