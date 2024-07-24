import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

# Loading the dataset
max_features = 10000 # Vocabulary Size
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# Inspect a sample review and its label 
sample_review = x_train[0] # Gives the one hot representation of every word in that record
sample_label = y_train[0]
print(sample_review, sample_label)

# Apply padding sequence
max_len = 500
x_train = sequence.pad_sequences(x_train, maxlen = max_len)
x_test = sequence.pad_sequences(x_test, maxlen = max_len)

# Train Rnn with embedding layers 
model = Sequential()
# Embedding layer to convert words in vectors of dimension 128
model.add(Embedding(max_features, 128, input_length=max_len))
model.add(SimpleRNN(128, activation = 'relu'))
model.add(Dense(1,activation='sigmoid'))
print(model.summary())   
model.compile(optimizer='adam', loss='binary_crossentropy', metrics='accuracy')

# Create an instance of Early Stopping Callback
from tensorflow.keras.callbacks import EarlyStopping
earlystopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2,callbacks=[earlystopping])
#Saving model
model.save('model_mapper.h5')
          





          
