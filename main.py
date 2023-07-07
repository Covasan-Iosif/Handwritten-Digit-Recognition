import os
import cv2 # computer vision - to load and process images
import numpy as np
import matplotlib.pyplot as plt # just for visualisation 
import tensorflow as tf


mnist = tf.keras.datasets.mnist

# split the data into training and testing data
# x - picture , y - the digit 
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# # normalizing the data (now the data is between 0 and 1)
# x_train  = tf.keras.utils.normalize(x_train, axis=1)
# x_test = tf.keras.utils.normalize(x_test, axis=1)

# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
# model.add(tf.keras.layers.Dense(128, activation='relu')) # Rectified Linear Unit
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(10, activation='softmax')) # the output layer having 10 individual digits (0-9)
# # with softmax all the outputs add up to 1 (the probability of each digit to be the right answear)

# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# # train the model , epoch = number of iterations
# model.fit(x_train, y_train, epochs=50)

# model.save('handwritten.model')

model = tf.keras.models.load_model('handwritten.model')

# loss, accuracy = model.evaluate(x_test, y_test)
# print(loss)
# print(accuracy)

image_number = 0
while os.path.isfile(f"digits/digit{image_number}.png"):
    try:
        img = cv2.imread(f"digits/digit{image_number}.png")[:,:,0] # we don't care about colors 
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"This digit is probably a {np.argmax(prediction)}") #argmax gives the index of the field with the highest number/ the neuron with the highest activation
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("Error!")
    finally: 
        image_number += 1