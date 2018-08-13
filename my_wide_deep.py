from keras.layers import Input, Dense, concatenate
from keras.models import Model
from keras.utils import plot_model

deep = Input(shape=(30,), name='input_deep')
wide = Input(shape=(30,), name='input_wide')

deep_dense_1 = Dense(8, activation='relu', name='deep_dense_1')(deep)
deep_dense_2 = Dense(30, activation='relu', name='deep_dense_2')(deep_dense_1)
deep_dense_3 = Dense(8, activation='relu', name='deep_dense_3')(deep_dense_2)

wide_deep = concatenate([deep_dense_3, wide])

ali = Dense(1, activation='sigmoid', name='wide_deep')(wide_deep)

end = Model(inputs=[deep, wide], outputs=ali)
