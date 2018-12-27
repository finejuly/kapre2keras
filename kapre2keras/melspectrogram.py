from keras.models import Sequential, Model
from keras.layers.convolutional import Conv2D
from keras.layers.core import Reshape, Dense, Permute, Lambda
from keras.layers import Add
import keras
import keras.backend as K

from kapre.time_frequency import Melspectrogram as kapre_Melspectrogram
import numpy as np

class Melspectrogram():
    def __init__(self, 
                 n_dft=512, 
                 n_hop=256, 
                 input_shape=(1,44100), 
                 padding='same', 
                 sr=16000, 
                 n_mels=128, 
                 fmin=0.0, 
                 fmax=8000, 
                 power_melgram=1.0, 
                 return_decibel_melgram=False, 
                 trainable_fb=False, 
                 trainable_kernel=False):
        
        self.n_dft=n_dft
        if self.n_dft%2==0:
            self.n_freq = self.n_dft/2+1
        else:
            self.n_freq = (self.n_dft+1)/2
        self.n_hop=n_hop
        self.input_shape=input_shape
        self.padding=padding
        self.sr=sr
        self.n_mels=n_mels
        self.fmin=fmin
        self.fmax=fmax
        self.power_melgram=float(power_melgram)
        self.return_decibel_melgram=return_decibel_melgram
        self.trainable_fb=trainable_fb
        self.trainable_kernel=trainable_kernel

    def __call__(self, inputs, mode='keras'):

        assert mode in ['keras', 'kapre'], "mode must be 'keras' (default) or 'kapre'"        
        
        layer_shape = Model(inputs=inputs, outputs=inputs).output_shape
        self.n_ch = layer_shape[1]
        self.n_sample = layer_shape[2]

        outputs = kapre_Melspectrogram(n_dft=self.n_dft, n_hop=self.n_hop, input_shape=self.input_shape, padding=self.padding, sr=self.sr, n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax, power_melgram=self.power_melgram, return_decibel_melgram=self.return_decibel_melgram, trainable_fb=self.trainable_fb, trainable_kernel=self.trainable_kernel)(inputs)
        
        if mode=='kapre':
            return outputs
        
        kapre_model = Model(inputs=inputs, outputs=outputs)
        self.w = kapre_model.get_weights()
        
        m = inputs
        m = Reshape((self.n_ch,-1,1))(m)
        m = Permute((2,1,3))(m)
        m_re = Conv2D(self.n_dft/2+1, kernel_size=(self.n_dft,1), strides=(self.n_hop, 1), 
                      padding=self.padding, use_bias=False, kernel_initializer=keras.initializers.Constant(value=self.w[0]), 
                      trainable=self.trainable_kernel, name='stft_real')(m)
        m_re = Lambda(lambda x: K.pow(x,2))(m_re)
        m_im = Conv2D(self.n_dft/2+1, kernel_size=(self.n_dft,1), strides=(self.n_hop, 1), 
                      padding=self.padding, use_bias=False, kernel_initializer=keras.initializers.Constant(value=self.w[1]), 
                      trainable=self.trainable_kernel, name='stft_imag')(m)
        m_im = Lambda(lambda x: K.pow(x,2))(m_im)
        m = Add()([m_re,m_im])
        m = Dense(self.n_mels, use_bias=False, kernel_initializer=keras.initializers.Constant(value=self.w[2]), trainable=self.trainable_fb, name='freq2mel')(m)
        if self.power_melgram!=2.0:
            m = Lambda(lambda x: K.pow(K.sqrt(x),self.power_melgram))(m)
        if self.return_decibel_melgram==True:
            amin=1e-10
            dynamic_range=80.0
            m = Lambda(lambda x: 10 * K.log(K.maximum(x, amin)) / np.log(10).astype(K.floatx()))(m)
            m = Lambda(lambda x: x - K.max(x))(m)
            m = Lambda(lambda x: K.maximum(x, -1 * dynamic_range) )(m)
            
        m = Permute((3,1,2))(m)
        outputs = m           

        return outputs