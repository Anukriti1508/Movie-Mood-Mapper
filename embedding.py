from tensorflow.keras.preprocessing.text import one_hot
sentences = ['the glass of milk',
        'the glass of juice',
        'the cup of tea',
        'I am a good boy',
        'I am a good developer',
        'understand the meaning of words',
        'your videos are good']
        
# Define Vocabulary Size
voc_size = 10000
# One Hot rep for every word
one_hot_repr = [one_hot(word, voc_size)for word in sentences]
print(one_hot_repr)

# Word embedding repr
from tensorflow.keras.layers import Embedding
from tensorflow.keras.utils import pad_sequences # To make sentences of equal size
from tensorflow.keras.models import Sequential
import numpy as np
# Set max sent length
sent_length = 8
embedded_docs = pad_sequences(one_hot_repr, padding = 'pre', maxlen = sent_length)
print(embedded_docs)

# Feature representation
dim = 10
model = Sequential()
model.add(Embedding(voc_size, dim, input_length=sent_length))
model.compile('adam','mse')
model.summary()
#Get the word embeddings
y = model.predict(embedded_docs)
print(y)







