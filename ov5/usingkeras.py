import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.optimizers import RMSprop
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import numpy as np

import scikitplot as skplt


data = pickle.load(open("keras-data.pickle", "rb"))

x_train = data["x_train"]
y_train = data["y_train"]
x_test = data["x_test"]
y_test = data["y_test"]
vocab_size = data["vocab_size"]
max_length = data["max_length"]

# Preprocess data
x_train = pad_sequences(x_train, maxlen=512)
x_test = pad_sequences(x_test, maxlen=512)
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)


# Init the model
model = Sequential()

# Add layers as specified in the assignment
model.add(Embedding(input_dim=vocab_size, output_dim=2))
model.add(LSTM(units=2))
model.add(Dense(units=2))


print("Compile time")
# Compile and train
model.compile(loss='mean_squared_error', optimizer='Adam', metrics=['accuracy'])

print("Model fitting")

history = model.fit(x_train, y_train, epochs=3, batch_size=128, verbose=1)

title = "Normalized confusion matrix for the model"

print("Model fitted")
y_predicted = model.predict_classes(x=x_test, batch_size=128)
skplt.metrics.plot_confusion_matrix(y_test, y_predicted, normalize=True, title=title)

# print(title)
# print(disp.confusion_matrix)
plt.show()


# THE FOLLOWING CODE IS COPIED FROM THE KERAS.IO WEBPAGE: https://keras.io/visualization/

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# Predict and evaluate
#pred = model.predict(x=padded_x_test, y=prep_y_test)
loss, accuracy = model.evaluate(x_test, y_test)
#plot_model(model, to_file='model.png')
print(model.summary())
print('Loss:\t', loss, '\nAccuracy:\t', accuracy)
