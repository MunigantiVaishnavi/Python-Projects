#import dependencies
import numpy
import sys
nltk.download('stopwords')
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from keras.models import Sequential
from keras.layers import Dense,Dropout,LSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
#load data
file=open("frankenstein-2.txt").read()
#tokenization
#standarization
def tokenize_words(input):
  input=input.lower()
  tokenizer=RegexpTokenizer(r'\w+')
  tokens=tokenizer.tokenize(input)
  filtered=filter(lambda token: token not in stopwords.words('english'),tokens)
  return " ".join(filtered)
processed_input=tokenize_words(file)
#chars to numbers
chars=sorted(list(set(processed_inputs)
char_to_num=dict((c,i) for i,c in enumerate(chars))
#check if words to chars or chars to num (?!) has worked?
input_len=len(processed_inputs)
vocal_len=len(chars)
print("total number of characters:",input_len)
print("total vocab:",vocab_len)
#seq length
seq_length+100
x_data=[]
y_data=[]
#loop through the sequence
for i in range(0, input_len - seq_length, l):
      #define i/p and o/p sequences
      #i/p is the current character plus the desired sequence length
      in_seq=processed_inputs[i:i + seq_length]
      #o/p is the initial character plus the desired sequence length
      out_seq=processed_inputs[i + seq_length]
      #converting the list of chars to integers based on previous values and appending the values to our lists
      x_data.append([char_to_num[char] for char in in_seq])
      x_data.append([char_to_num[out_seq])
#check to see how many total i/p sequences we have 
n_patterns=len(x_data)
print("total patterns:",n_patterns)
#convert i/p sequence to np array that our network can use
X=numpy.reshape(x_data,(n_patterns, seq_length, 1))
X=X/float(vocab_len)
#one-hot encoding our label data
y=np_utils.to_categorical(y_data)
# creating the model
#creating a sequential model
#dropout is used to prevent overfitting
model=Sequential()
model.add(LSTM(256, input_shape=(X.shape[1],X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256,return_sequences=True))
model.add(Dropout)
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1],activation='softmax'))
#compile the model
model.compile(loss='categorical_crossentropy',optimizer='adam')
#saving weights
filepath="model_weights_saved.hdf5"
checkpoint= ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
desired_callbacks=[checkpoint]
#fit model and let it train
model.fit(X,y, epochs=4, batch_size=256, callbacks=desired_callbacks)
#recompile model with the saved weights
filename='model_weights_saved.hdf5'
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam)
#output of the model back into characters
num_to_char=dict((i,c) for i,c in enumeration(chars))
#random seed to help generate
start=numpy.random.randint(0, len(X_data) -1)
pattern=X_data[start]
print("random seed:")
print("\"", "".join((num_to_char[value] for value in pattern]),"\"")
#generate the text
for i in range(1000):
  x=numpy.reshape(pattern, (1,len(pattern), 1))
  x=x/float(vocab_len)
  prediction=model.predict(x, verbose=0)
  index=numpy.argmax(prediction)
  result=num_to_char[index]
  seq_in=[num_to_char[value] for value in pattern]
  sys.stdout.write(result)
  pattern.append(index)
  pattern=pattern[1:len(pattern)]
