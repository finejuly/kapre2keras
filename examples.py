import numpy as np

from keras.layers import Input
from keras.models import Model

from kapre2keras.melspectrogram import Melspectrogram
 
sr = 44100
n_ch = 2
n_sample = 44100
input_shape = (n_ch, n_sample)

inputs = Input(shape=(n_ch,None))

outputs = Melspectrogram(n_dft=512, n_hop=256, input_shape=input_shape, 
                         padding='same', sr=sr, n_mels=128, fmin=0.0, fmax=sr/2, 
                         power_melgram=1.0, return_decibel_melgram=False, 
                         trainable_fb=False, trainable_kernel=False)(inputs, mode='keras') # mode='kapre' if you want to get the original custom kapre layer.

model = Model(inputs=inputs, outputs=outputs)
model.summary()

x = np.random.random((1,n_ch,n_sample))
mel = model.predict(x)

model.save('aa.h5')
