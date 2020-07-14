import numpy as np
import pandas as pd
import pickle
import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from CAE import _define_model

## Load Data
data_path = 'Y:/1. 프로젝트/2020_현대자동차_DS/2020_가속내구시험팀/과제5 표준가이드/data/20200630/A_Seg/train/'
A_Seg_data_list = os.listdir(data_path)

aseg_dat = []
for file in A_Seg_data_list:
    tmp_dat = pd.read_csv(os.path.join(data_path,file)).drop('Time', axis=1)
    aseg_dat.append(tmp_dat)

max_len = 0
min_len = 16000000
for a in aseg_dat:
    if max_len < len(a):
        max_len = len(a)
    if min_len > len(a):
        min_len = len(a)

padded_aseg_dat = []
for a in aseg_dat:
    pad = pd.DataFrame(0, index=np.arange(len(a),max_len), columns=a.columns)
    padded_aseg_dat.append(a.append(pad).reset_index(drop=True))

X = np.asarray([data_frame.to_numpy() for data_frame in padded_aseg_dat])
X = np.expand_dims(X,2)
hidden_size = [16]

for h in hidden_size:
    with tf.device('/CPU:0'):
        opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
        ae = _define_model(window_size=23742, num_channels=24,
                         filter_size=20, filter_dim=20,
                         embedding_dim=h,
                         kernel_initializer='glorot_normal',
                         kernel_regularizer=None)
        # es = [EarlyStopping(monitor='val_loss', patience=10)]
        ae.compile(optimizer=opt, loss='mse', metrics=['mse'])
        ae.fit(X, X, batch_size=8, epochs=200) # , validation_split = 0.1, callbacks=es
    ae.save_weights("model/0713_CAE_Model_{}.h5".format(h))

