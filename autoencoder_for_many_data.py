import matplotlib.pyplot as plt
import librosa
import librosa.display
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras import backend as K
import numpy as np
import os

def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train','validation'],loc='upper left')
    plt.show()

train_data = np.load("train_data.npy")

#---Define the model---
input_img = Input(shape=(train_data.shape[1],train_data.shape[2],train_data.shape[3]))
print(input_img.shape)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
print(x.shape)
x = MaxPooling2D((2, 2), padding='same')(x)
print(x.shape)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
print(x.shape)
x = MaxPooling2D((2, 2), padding='same')(x)
print(x.shape)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
print(x.shape)
encoded = MaxPooling2D((2, 2), padding='same')(x)
print(encoded.shape) #(None,16,6,8)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
print(x.shape)
x = UpSampling2D((2, 2))(x)
print(x.shape)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
print(x.shape)
x = UpSampling2D((2, 2))(x)
print(x.shape)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
print(x.shape)
x = UpSampling2D((2, 2))(x)
print(x.shape)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
print(decoded.shape)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy',metrics=['acc'])

#---training step---
train_history = autoencoder.fit(x=train_data,y=train_data,batch_size=112,validation_split=0.5,epochs=50,verbose=2)
show_train_history(train_history,'acc','val_acc')
show_train_history(train_history,'loss','val_loss')
autoencoder.save("my_model_for_many_data.h5")
