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

y,sr=librosa.load("speechdata/training/N110014.wav")
if(y.shape[0]<24064):
    y=np.pad(y,(0,24064-y.shape[0]),"constant",constant_values=(0,0))
elif(y.shape[0]>24064):
    y=y[0:24064]
melspec = librosa.feature.melspectrogram(y=y,sr=sr,n_fft=1024,hop_length=512,n_mels=128)
log_melspec = librosa.amplitude_to_db(melspec)
minimum = log_melspec.min() 
maximum = log_melspec.max()
img = scale_minmax(log_melspec)
img = img.reshape(img.shape[0],img.shape[1],1)
img = img.reshape(1,img.shape[0],img.shape[1],img.shape[2])

#reverse decoded array to audio
autoencoder = load_model("my_model_for_many_data.h5")
decoded_imgs = autoencoder.predict(img) 
print(decoded_imgs)
print(" ")
output_imgs = np.zeros((128,48))
for i in range (decoded_imgs.shape[1]):
    for j in range (decoded_imgs.shape[2]):
        output_imgs[i][j] = decoded_imgs[0][i][j][0]
print(output_imgs)
print(" ")
output_imgs = scale_minmax(output_imgs,minimum,maximum)
print(output_imgs)
print(" ")
output_melspec = librosa.db_to_amplitude(output_imgs)
print(output_melspec)
print(" ")
output_audio = librosa.feature.inverse.mel_to_audio(output_melspec) #inverse_mel_to_stft and then griffinlim
print(output_audio.shape)

#show as image
plt.figure()
ax = plt.subplot(2,1,1)
librosa.display.waveplot(y,sr=sr,color='b')
plt.title("original")
plt.xlabel("")
plt.subplot(2,1,2,sharex=ax,sharey=ax)
librosa.display.waveplot(output_audio,sr=sr,color='g')
plt.title("reconstruct")
plt.tight_layout()
plt.show()
librosa.output.write_wav("reconstruct.wav",output_audio,sr)

