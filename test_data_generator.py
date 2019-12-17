import matplotlib.pyplot as plt
import librosa
import librosa.display
import keras
from keras.models import Sequential, Model
from keras.models import load_model
from keras import backend as K
import numpy as np
import os

def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

test_dir="speechdata/testing"
test_list=os.listdir(test_dir)
test_list_len = len(test_list)

test_data = np.empty([test_list_len,128,48,1])

i=0
for f in test_list:
    y,sr=librosa.load(os.path.join(test_dir,f))
    if(y.shape[0]<24064):
        y=np.pad(y,(0,24064-y.shape[0]),"constant",constant_values=(0,0))
    elif(y.shape[0]>24064):
        y=y[0:24064]
    melspec = librosa.feature.melspectrogram(y=y,sr=sr,n_fft=1024,hop_length=512,n_mels=128)
    log_melspec = librosa.amplitude_to_db(melspec)
    img = scale_minmax(log_melspec)
    img = img.reshape(img.shape[0],img.shape[1],1)
    test_data[i] = img
    i = i+1

np.save("test_data",test_data)

