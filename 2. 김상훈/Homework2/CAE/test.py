import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from CAE import _define_model
from pathlib import Path
from itertools import chain, repeat
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras import Sequential, Model, regularizers
from sklearn.manifold import TSNE
from sklearn import preprocessing

## Functions
def extract_layers(main_model, starting_layer_ix, ending_layer_ix):
    if starting_layer_ix != 0:
        new_model = Sequential()
        new_model.add(Input(shape=main_model.get_layer(index=starting_layer_ix).input.shape[1], name='input'))
    else:
        new_model = Sequential()
    # create an empty model
    for ix in range(starting_layer_ix, ending_layer_ix + 1):
        curr_layer = main_model.get_layer(index=ix)
        # copy this layer over to the new model
        new_model.add(curr_layer)
    return new_model

def bayesian_model(h):
    inputs = Input(shape=(len(train_spec.keys())))
    x = Dense(h, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(inputs)
    x = Dropout(0.3)(x,training = True)
    x = Dense(h, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
    x = Dropout(0.3)(x, training=True)
    outputs = Dense(h)(x)
    model = Model(inputs, outputs)
    optimizer = tf.keras.optimizers.Adam(lr = 0.0001, decay=0.00001)
    model.compile(loss='mse',optimizer=optimizer, metrics=['mae', 'mse'])
    return model

def linear_model(h):
    with tf.device('/CPU:0'):
        model = Sequential([
        Dense(h, activation='relu', input_shape=[len(train_spec.keys())]),
        Dropout(0.3),
        Dense(h, activation='relu'),
        Dropout(0.3),
        Dense(h, activity_regularizer='l2')])
        optimizer = tf.keras.optimizers.RMSprop(0.001)
        model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    return model
def get_velocity(f):
    time = f['Time'][-1:].values[0] - f['Time'][0]
    return int(round(800/time*3600/1000,0))

class PrintDot(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')
      
def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure(figsize=(8,12))

    plt.subplot(2,1,1)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mae'],
           label='Train Error')
    plt.plot(hist['epoch'], hist['val_mae'],
           label = 'Val Error')
    plt.legend()

    plt.subplot(2,1,2)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mse'],
           label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'],
           label = 'Val Error')
    plt.legend()
    plt.show()

## Path
base_path = 'C:/Users/Sanghoon/Desktop/Github/Task5_StandardGuide/CAE/'
data_path = 'Y:/1. 프로젝트/2020_현대자동차_DS/2020_가속내구시험팀/과제5 표준가이드/data/20200630/A_Seg/train/'

save_path = os.path.join('Y:/1. 프로젝트/2020_현대자동차_DS/2020_가속내구시험팀/과제5 표준가이드/notsplit_results/')
save_path = Path(save_path)
save_path.mkdir(parents=True, exist_ok=True)

model_path = os.path.join(base_path,'model')
model_path = Path(model_path)
model_path.mkdir(parents=True, exist_ok=True)

## Load Data
A_Seg_data_list = os.listdir(data_path)
aseg_dat = []
spec_indicator = []
velocity = []
for file in A_Seg_data_list:
    spec_indicator.append(file.split('A_Seg_')[1].split('_2UP')[0])
    tmp_dat = pd.read_csv(os.path.join(data_path,file))
    velocity.append(get_velocity(tmp_dat))
    tmp_dat = tmp_dat.drop('Time', axis=1)
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

# Spec data
spec_path = '//163.152.174.30/NewShare/1. 프로젝트/2020_현대자동차_DS/2020_가속내구시험팀/과제5 표준가이드/data//과제5_20200508/'
tmp_spec = pd.read_csv(os.path.join(spec_path, 'A_seg_reference2.csv'), encoding='cp949')
tmp_spec['Unnamed: 0'] = [file_name.split('_2UP')[0] for file_name in tmp_spec['Unnamed: 0']]
tmp_spec = tmp_spec.set_index('Unnamed: 0')

spec_data = []
for car in spec_indicator:
    spec_data.append(tmp_spec.loc[car].to_numpy())
velocity = np.asarray(velocity)
spec_data = np.asarray(spec_data)

s = pd.DataFrame(spec_data, columns=['축중량_FRT', '축중량_RR', '휠베이스', '휠트레드_FRT', '휠트레드_RR'])
s['velocity'] = velocity
s.to_csv(os.path.join(save_path,"Spec_data.csv"), index=False)
pd.DataFrame(spec_indicator).to_csv(os.path.join(save_path,"Car_name_data.csv"), index=False)

le = preprocessing.LabelEncoder()
le.fit(spec_indicator)
car_class = le.transform(spec_indicator)

ch_names = ['FR LH Fx', 'FR LH Fy', 'FR LH Fz', 'FR LH Mx', 'FR LH My', 'FR LH Mz',
               'FR RH Fx', 'FR RH Fy', 'FR RH Fz', 'FR RH Mx', 'FR RH My', 'FR RH Mz',
               'RR LH Fx', 'RR LH Fy', 'RR LH Fz', 'RR LH Mx', 'RR LH My', 'RR LH Mz',
               'RR RH Fx', 'RR RH Fy', 'RR RH Fz', 'RR RH Mx', 'RR RH My', 'RR RH Mz']
CAE_model = _define_model(window_size=23742, num_channels=24,
                    filter_size=20, filter_dim=20,
                    embedding_dim=16,
                    kernel_initializer='glorot_normal',
                    kernel_regularizer=None)

## Test_CAE_Model
hidden_size = [16]
for h in hidden_size:
    CAE_model = _define_model(window_size=23742, num_channels=24,
                    filter_size=20, filter_dim=20,
                    embedding_dim=h,
                    kernel_initializer='glorot_normal',
                    kernel_regularizer=None)

    CAE_model.load_weights(os.path.join(model_path,"0713_CAE_Model_{}.h5".format(h)))

## Split Encoder & Decoder
    encoder_model = extract_layers(CAE_model, 0, 13)
    decoder_model = extract_layers(CAE_model, 14, 29)

    with tf.device('/CPU:0'):
        hidden_vector = encoder_model.predict(X)

    target_hidden = pd.DataFrame(hidden_vector)
    target_hidden.to_csv(os.path.join(save_path,"0713_Hidden_Vector_{}.csv".format(h)), index=False, header=False)

    model = TSNE(learning_rate=100)
    transformed = model.fit_transform(target_hidden)

    xs = transformed[:, 0]
    ys = transformed[:, 1]
    plt.scatter(xs, ys, c = car_class)
    plt.savefig(os.path.join(save_path,"0713_Cluster_Hiddensize_{}.png".format(h)))
    plt.close()

    input_spec = s
    train_spec = input_spec.sample(frac=0.8,random_state=0)
    test_spec = input_spec.drop(train_spec.index)
    train_hidden = target_hidden.iloc[train_spec.index]
    test_hidden = target_hidden.drop(train_spec.index)

    model = bayesian_model(h)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)

    EPOCHS = 1000
    with tf.device('/CPU:0'):
        history = model.fit(train_spec, train_hidden, epochs=EPOCHS, validation_split=0.2, verbose=0, callbacks=[early_stop, PrintDot()])
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    with tf.device('/CPU:0'):
        pred_hidden = model.predict(test_spec)
        decoder_output = decoder_model.predict(pred_hidden)
        true_decoder_out = decoder_model.predict(target_hidden)

    pd.DataFrame(pred_hidden).to_csv(os.path.join(save_path,"0713_Predict_Hidden_Vector_{}.csv".format(h)), index=False, header=False)

    decoder_output = decoder_output.transpose(2,0,1,3)
    decoder_output = np.squeeze(decoder_output, axis=0)
    X_transposed = X.transpose(2,0,1,3)
    X_frame = np.squeeze(X_transposed, axis=0)

    true_decoder_out = true_decoder_out.transpose(2, 0, 1, 3)
    true_decoder_out = np.squeeze(true_decoder_out, axis=0)

    train_index = list(train_spec.index)

    result_path = os.path.join(save_path, 'maxpooling_hiddensize_{}'.format(h))
    result_path = Path(result_path)
    result_path.mkdir(parents=True, exist_ok=True)

    complement = list(set(list(range(0, len(X_frame)))).difference(train_index))
    for i,a in enumerate(complement):
        pd.DataFrame(X_frame[a], columns=ch_names).to_csv(os.path.join(result_path,"0713_True_{}.csv".format(i)), index=False)
    for i in range(len(decoder_output)):
        pd.DataFrame(decoder_output[i], columns=ch_names).to_csv(os.path.join(result_path,"0713_Inference_{}.csv".format(i)), index=False)
    for i in range(len(true_decoder_out)):
        pd.DataFrame(true_decoder_out[i], columns=ch_names).to_csv(os.path.join(result_path,"0713_decoder_{}.csv".format(i)), index=False)
    for i in range(len(X_frame)):
        pd.DataFrame(X_frame[i], columns=ch_names).to_csv(os.path.join(result_path,"0713_Target_all_{}.csv".format(i)), index=False)
    for i in train_index:
        pd.DataFrame(X_frame[i], columns=ch_names).to_csv(os.path.join(result_path, "0713_Training_inference_{}.csv".format(i)), index=False)






