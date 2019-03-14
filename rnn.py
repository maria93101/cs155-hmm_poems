



from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import io
import re
import string


def load_poems(filename):

    lines = [] # 2d dictionary, each array is a split + cleaned line
    words = {} # dictionary of a word, and its frequency

    file = open(filename, 'r')

    for line in file:
        line = line.strip()
        if  len(line) < 10:
            # Too short to be a valid line
            continue
        line = "".join(l for l in line if l not in string.punctuation)
        line = line.lower()
        line = line.split()

        lines.append(line)

        for word in line:
            try:
                # add to frequency if the word is already in the dic
                words[word] += 1
            except KeyError:
                # if not, add the word to the dic
                words[word] = 1
    return lines, words

file = "data/shakespeare.txt"
lines, words = load_poems(file)

text = ""
for i in lines:
    text += " ".join(i)+"\n"


chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])

x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# build the model: a single LSTM
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars), activation='softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

model.fit(x, y,
          batch_size=32,
          epochs=500)

model.save('500_epoch_1_step_32_batch_rnn_model.h5')
