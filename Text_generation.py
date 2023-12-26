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
      in_seq=processed_inputs[i:i + seq_length]
      out_seq=processed_inputs[i + seq_length]
      x_data.append([char_to_num[char] for char in in_seq])
      x_data.append([char_to_num[out_seq])
n_patterns=len(x_data)
print("total patterns:",n_patterns)
X=numpy.reshape(x_data,(n_patterns, seq_length, 1))
X=X/float(vocab_len)
#one-hot encoding
y=np_utils.to_categorical(y_data)
# creating the model
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
        
