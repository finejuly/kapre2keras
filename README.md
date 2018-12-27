# kapre2keras
Represent Kapre Melspectrogram layer as conventional Keras layers.

# why?

Although Kapre (https://github.com/keunwoochoi/kapre) is awesome, it is a bit cumbersome for Android or Raspberry pi because it uses custom layer. kapre2keras simply represents this custom layer by using conventional Keras layers.

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

# Contact

Il-Young Jeong @ cochlear.ai
iyjeong@cochlear.ai
