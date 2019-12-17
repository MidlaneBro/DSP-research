import matplotlib.pyplot as plt
import librosa
import librosa.display
import keras
from keras.models import Sequential, Model
from keras.models import load_model
from keras import backend as K
import numpy as np
import os

test_data = np.load("test_data.npy")

#reverse decoded array to audio
autoencoder = load_model("my_model_for_many_data.h5")
loss,accuracy = autoencoder.evaluate(x=test_data,y=test_data,verbose=1)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss,accuracy*100))

