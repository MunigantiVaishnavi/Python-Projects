#import dependencies
import numpy
import sys
import nltk
nltk.download('stopwords')
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

#load data
#loading data and opening our input data in the form of a text file
#project gutenburg/berg is where the data can be found(google it)
file=open("Shelley-1818 Frankenstein.txt").read()

#tokenization and standardization
#what is tokenization? it is the process of breaking a stream of text up into words phrases symbols or other meaningful elements
stop_words = set(stopwords.words('english'))  # Precompute stopwords for performance
def tokenize_words(input):
    #lowercase everything to standaridize it
    input = input.lower()
    #instantiating the tokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    #tokenizing the text into tokens
    tokens = tokenizer.tokenize(input)
    #filtering the stopwords using lambda
    filtered = [token for token in tokens if token not in stop_words]
    return " ".join(filtered)

#preprocess the input datamake tokens
processed_input = tokenize_words(file)

#chars to numbers
#convert characters in our input to numbers
#we will sort the list of the set of all characters that appear in out i/p text and then use the enumerate fn to get numbers that represent the characters
#we will then create a dictionary that stores the keys and values, or the characters and the numbers that represent them
chars = sorted(list(set(processed_input)))  # Corrected the variable name
char_to_num = dict((c, i) for i, c in enumerate(chars))

#check if words to chars or chars to num (?!) has worked?
#just so we get an idea of whether our process of coverting words to characters has worked
#we print the length of our variables
input_len = len(processed_input)
vocal_len = len(chars)  # Changed to vocal_len
print("total number of characters:", input_len)
print("total vocab:", vocal_len)

#seq length
#we are defining how log we want an individual sequence  here
#an individual sequence is a complete mapping of input characters as integers 
seq_length = 100  # Fixed seq_length assignment
x_data = []
y_data = []

#loop through the sequence
for i in range(0, input_len - seq_length, 1):
    in_seq = processed_input[i:i + seq_length]
    out_seq = processed_input[i + seq_length]
    x_data.append([char_to_num[char] for char in in_seq])
    y_data.append(char_to_num[out_seq])  # Fixed list append issue

#check to see how many total i/p sequences we have
n_patterns = len(x_data)
print("total patterns:", n_patterns)

#convert i/p sequence to np array
X = numpy.reshape(x_data, (n_patterns, seq_length, 1))
X = X / float(vocal_len)

#one-hot encoding our label data
y = np_utils.to_categorical(y_data)

# creating the model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.2))  # Fixed Dropout layer
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))

#compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam')

#saving weights
filepath = "model_weights_saved.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
desired_callbacks = [checkpoint]

#fit model and let it train
model.fit(X, y, epochs=4, batch_size=256, callbacks=desired_callbacks)

#recompile model with the saved weights
filename = 'model_weights_saved.hdf5'
model.load_weights(filename)

#output of the model back into characters
num_to_char = dict((i, c) for i, c in enumerate(chars))

#random seed to help generate
start = numpy.random.randint(0, len(x_data) - 1)
pattern = x_data[start]
print("random seed:")
print("\"", "".join((num_to_char[value] for value in pattern)), "\"")

#generate the text
for i in range(1000):
    x = numpy.reshape(pattern, (1, len(pattern), 1))
    x = x / float(vocal_len)
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    result = num_to_char[index]
    sys.stdout.write(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]  # Keep the pattern length constant
