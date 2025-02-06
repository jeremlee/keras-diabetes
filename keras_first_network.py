from numpy import loadtxt  
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import Dense, Input
# load the dataset  

dataset = loadtxt('pima-indians-diabetes.data.csv', delimiter=',')  

# split into input (X) and output (y) variables  

X = dataset[:,0:8]  
y = dataset[:,8]  

# define the keras model  

model = Sequential()  
# creating the input layer in another line rather than putting it with the first hidden layer
model.add(Input(shape=(8,)))  
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation='relu'))  
model.add(Dense(1, activation='sigmoid'))  

# compile the keras model  

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  

# fit the keras model on the dataset  

model.fit(X, y, epochs=150, batch_size=10)  

# evaluate the keras model  

_, accuracy = model.evaluate(X, y)  
print('Accuracy: %.2f' % (accuracy*100))

# make class predictions with the model  

predictions = (model.predict(X) > 0.5).astype(int)

for i in range(5):  
    print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))  