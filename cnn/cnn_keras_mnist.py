
# coding: utf-8

# # CNN with Keras 
# 
# ## Import Library and Load MNIST Data


import keras
from keras.datasets import mnist
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
import matplotlib.pylab as plt


# We are using tensorflow-gpu so its best to test if its working
from keras import backend as K
K.tensorflow_backend._get_available_gpus()

#Load the MNIST dataset.
(X, y), (X_test,y_test) = mnist.load_data()


# # Define HyperParameters and Initialize them

#We will need to tune this hyperparameter for best result. 
num_classes = 10
epochs = 10
batch_size = 128


# # Preprocess the training data

# ## Reshape
# Intialize height and width of the image. In case of MNIST data its 28x28
# 
# Reshape the MNIST data into 4D tensor (no_of_sample, width, height, channels)
# 
# MNIST image is in grayscale so channels will be 1 in our case.
# 
# ## Convert the data into right type
# 
# Convert the data into float.
# 
# Divide it by 255 to normalize. Since color ranges from 0 to 255. 
# 

#Initialize variable
width = 28
height = 28
no_channel = 1
input_shape = (width,height,no_channel)

#Reshape input
X = X.reshape(X.shape[0],width,height,no_channel)
X_test = X_test.reshape(X_test.shape[0],width,height,no_channel)

#Convert to float
X = X.astype('float32')
X_test = X_test.astype('float32')
                         
#Normalize
X = X/255
X_test = X_test/255   

print('x_train shape:', X.shape)
print(X.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
print(y.shape, 'output shape')


# # Convert Output to one hot vector.
# 
# For example 3 would be represented by 1 at 3rd index of ten zeros where 1st one reperesent 0 and last 9. 
# 
# 0001000000

y = keras.utils.to_categorical(y,num_classes=10)
y_test = keras.utils.to_categorical(y_test,num_classes=10)

# # Define Keras Model and stack the layers 
# 
# 
# This will have Conv Layer followed with relu activation. MaxPooling will be applied after that to subsample. 
# 

#Define model
model = Sequential()

#Layer1 = Conv + relu + maxpooling
model.add(Conv2D(filters=32,kernel_size=(5,5),strides=(1,1),padding='same',activation='relu',input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))

#Layer2 = Conv + relu + maxpooling
model.add(Conv2D(filters=64,kernel_size=(5,5),strides=(1,1),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

#Flatten
model.add(Flatten())

#Layer Fully connected
model.add(Dense(units=1000,activation='relu'))
model.add(Dense(units=num_classes,activation='softmax'))

#Compile model
model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.categorical_crossentropy, 
              metrics=['accuracy'])


# # Optional class for saving accuracy history
# 
# We will need the accuracy at each epoch to plot the graph

class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))

history = AccuracyHistory()        


# # Train the model

model.fit(x=X,
          y=y,
          batch_size=batch_size,
          epochs=10,
          verbose=1,
          validation_data=(X_test,y_test), 
          callbacks=[history])


# # Model Evaluation
testScore = model.evaluate(x=X_test,y=y_test,verbose=1)
print('Test loss:', testScore[0])
print('Test accuracy:', testScore[1])


# # Plot epoch vs accuracy.
plt.plot(range(1,11),history.acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
