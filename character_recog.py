# organize imports
from keras.models import Sequential
from keras.layers import Dense
#from keras.callbacks import LambdaCallback

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from keras.models import model_from_json

# seed for reproducing same results
seed =2
np.random.seed(seed)
names = [i for i in range(785)]
dataset = pd.read_csv('C:\\Users\\PAV CC102TX\\Downloads\\images.csv', names=names)
X = dataset.loc[:,1:]

Y = dataset.loc[:,0]
enc = OneHotEncoder()
Y = enc.fit_transform(np.array(Y).reshape(-1,1))
#print(Y.toarray())


# split the data into training (67%) and testing (33%)
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.33, random_state=seed)
#(X_train, X_val,Y_train, Y_val) = train_test_split(X_train,Y_train, test_size=0.33,random_state=seed)
# create the model
model = Sequential()
model.add(Dense(256, input_dim=784, init='uniform', activation='relu'))
model.add(Dense(256, init='uniform', activation='relu'))
model.add(Dense(256, init='uniform', activation='relu'))

model.add(Dense(26, init='uniform', activation='relu'))

model.add(Dense(26, init='uniform', activation='softmax'))

# compile the model
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

#fit the model
model.fit(X_train, Y_train, validation_data=(X_test, Y_test),
          epochs=10, batch_size=5, verbose=0)
#print(X_train,Y_train)
#print(X_test,Y_test)
# evaluate the model
scores = model.evaluate(X_test, Y_test)
print("Accuracy: %.2f%%" % (scores[1]*100))
