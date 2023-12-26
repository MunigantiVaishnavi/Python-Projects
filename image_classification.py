import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
%matplotlib inline
#usig pandas to read the database stored in the same folder
data=pd.read_csv('mnist.csv')
#viewing the column heads
data.head()
#extracting data from the dataset and viewing them up close
a=data.iloc[3,1:].values
#reshaping the extracted data into a reasonable size
a=a.reshape(28,28).astype('uint8')
plt.imshow(a)
#preparing the data
#aeparating labels and data values
df_x=data.iloc[:,1:]
df_y=data.iloc[:,0]
#creating test and train sizes
x_train, x_test, y_train, y_test= train_test_split(df_x,df_y,test_size=0.2, random_state=4)
#check data
y_train.head()
#callig rf classifier
rf=RandomForestClassifier(n_estimators=100)
#fit the model
rf.fit(x_train. y_train)
#prediction on test data
pred
#check prediction accuracy
s=y_test.values
#calculate no of correctly predicted values
count=0
for i in range(len(pred)):
	if pred[i]==s[i]:
		count=count+1
count
#total values that the prediction code was running on
len(pred)
#acuuracy value
pred/count

pred=rf.predict(x_test) 