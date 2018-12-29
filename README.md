# kapre2keras
Represent Kapre Melspectrogram layer as conventional Keras layers.

# why?

Although Kapre (https://github.com/keunwoochoi/kapre) is awesome, it is a bit cumbersome for Android or Raspberry pi because it uses custom layer and requires librosa. kapre2keras simply represents this custom layer by using conventional Keras layers.

# Usage

- For the details about the parameters, please refer Kapre.

```
from kapre2keras.melspectrogram import Melspectrogram

outputs = Melspectrogram(
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
                 trainable_kernel=False)(inputs, mode='keras')
```

# Outputs

## model.summary() with mode='kapre'

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2, None)           0         
_________________________________________________________________
melspectrogram_1 (Melspectro (None, 128, None, 2)      296064    
=================================================================
Total params: 296,064
Trainable params: 0
Non-trainable params: 296,064
_________________________________________________________________
```

## model.summary() with mode='keras'
```
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            (None, 2, None)      0                                            
__________________________________________________________________________________________________
reshape_1 (Reshape)             (None, 2, None, 1)   0           input_1[0][0]                    
__________________________________________________________________________________________________
permute_1 (Permute)             (None, None, 2, 1)   0           reshape_1[0][0]                  
__________________________________________________________________________________________________
stft_real (Conv2D)              (None, None, 2, 257) 131584      permute_1[0][0]                  
__________________________________________________________________________________________________
stft_imag (Conv2D)              (None, None, 2, 257) 131584      permute_1[0][0]                  
__________________________________________________________________________________________________
lambda_1 (Lambda)               (None, None, 2, 257) 0           stft_real[0][0]                  
__________________________________________________________________________________________________
lambda_2 (Lambda)               (None, None, 2, 257) 0           stft_imag[0][0]                  
__________________________________________________________________________________________________
add_1 (Add)                     (None, None, 2, 257) 0           lambda_1[0][0]                   
                                                                 lambda_2[0][0]                   
__________________________________________________________________________________________________
freq2mel (Dense)                (None, None, 2, 128) 32896       add_1[0][0]                      
__________________________________________________________________________________________________
lambda_3 (Lambda)               (None, None, 2, 128) 0           freq2mel[0][0]                   
__________________________________________________________________________________________________
permute_2 (Permute)             (None, 128, None, 2) 0           lambda_3[0][0]                   
==================================================================================================
Total params: 296,064
Trainable params: 0
Non-trainable params: 296,064
__________________________________________________________________________________________________
```

# Requirements

- Model generation phase:
Kapre (and Librosa and Keras for it)

- Training and inference phase:
Keras

# Contact

Il-Young Jeong @ cochlear.ai
iyjeong@cochlear.ai
