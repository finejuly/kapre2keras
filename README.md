# kapre2keras
Converts Kapre Melspectrogram layer to Keras representation.

# why?

Although Kapre (https://github.com/keunwoochoi/kapre) is awesome, it is a bit cumbersome for Android, Raspberry pi because it uses custom layer. kapre2keras simply converts this custom layer to the conventional keras layers.

# ---

Current version does not support 'return_decibel_melgram' and it will always set to be 'False'.
