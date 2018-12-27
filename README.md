# kapre2keras
Converts Kapre Melspectrogram layer to Keras representation.

# why?

Although Kapre (https://github.com/keunwoochoi/kapre) is awesome, it is a bit cumbersome for Android, Raspberry pi because it uses custom layer. kapre2keras simply converts this custom layer to the conventional keras layers.

# Usage

- For the details about parameters, please refer Kapre.

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
