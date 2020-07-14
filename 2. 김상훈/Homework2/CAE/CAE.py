from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import MaxPooling2D

def ConvBNActBlock(x, activation, filter_size, filter_dim, kernel_initializer, kernel_regularizer, name=None):
    x = Conv2D(filters=filter_dim,
               kernel_size=filter_size,
               padding='same',
               kernel_initializer=kernel_initializer,
               kernel_regularizer=kernel_regularizer,
               name=f'{name}-conv')(x)
    x = BatchNormalization(name=f'{name}-bn')(x)
    x = Activation(activation, name=f'{name}-act')(x)

    return x


def ConvTrBNActBlock(x, activation, filter_size, filter_dim, kernel_initializer, kernel_regularizer, name=None):
    x = Conv2DTranspose(filters=filter_dim,
                        kernel_size=filter_size,
                        padding='same',
                        kernel_initializer=kernel_initializer,
                        kernel_regularizer=kernel_regularizer,
                        name=f'{name}-convtr')(x)
    x = BatchNormalization(name=f'{name}-bn')(x)
    x = Activation(activation, name=f'{name}-act')(x)

    return x


def BottleneckBlock(x, activation, embedding_dim, kernel_initializer, kernel_regularizer, name=None):
    _raw_shape = x.shape[1:]
    x = Flatten(name=f'{name}-flatten')(x)
    _flat_shape = x.shape[1]

    h = Dense(units=embedding_dim,
              kernel_initializer=kernel_initializer,
              kernel_regularizer=kernel_regularizer,
              name=f'{name}-embedding')(x)
    x = BatchNormalization(name=f'{name}-bn1')(h)
    x = Activation(activation, name=f'{name}-act1')(x)

    x = Dense(units=_flat_shape,
              kernel_initializer=kernel_initializer,
              kernel_regularizer=kernel_regularizer,
              name=f'{name}-dense')(x)
    x = Reshape(_raw_shape, name=f'{name}-reshape')(x)
    x = BatchNormalization(name=f'{name}-bn2')(x)
    x = Activation(activation, name=f'{name}-act2')(x)

    return h, x


def _define_model(window_size, num_channels,
                  filter_size, filter_dim,
                  embedding_dim,
                  kernel_initializer,
                  kernel_regularizer,
                  activation='relu',
                  output_act=None):
    filter_size = (filter_size, 1)

    conv_block_args = {
        'activation': activation,
        'filter_size': filter_size,
        'kernel_initializer': kernel_initializer,
        'kernel_regularizer': kernel_regularizer
    }

    _x_in = Input(shape=(window_size, 1, num_channels), name='input')

    x = ConvBNActBlock(_x_in, name='block1', filter_dim=filter_dim, **conv_block_args)
    #x = AveragePooling2D(pool_size=(2, 1), name='pool1')(x)
    x = MaxPooling2D(pool_size=(2, 1), name='pool1')(x)
    x = ConvBNActBlock(x, name='block2', filter_dim=filter_dim * 2, **conv_block_args)
    #x = AveragePooling2D(pool_size=(3, 1), name='pool2')(x)
    x = MaxPooling2D(pool_size=(3, 1), name='pool2')(x)
    x = ConvBNActBlock(x, name='block3', filter_dim=filter_dim * 4, **conv_block_args)

    _z_out, x = BottleneckBlock(x, activation, embedding_dim, kernel_initializer,
                                kernel_regularizer, name='blottleneck')

    x = UpSampling2D(size=(3, 1), name='upsample1')(x)
    x = ConvTrBNActBlock(x, name='block4', filter_dim=filter_dim * 2, **conv_block_args)

    x = UpSampling2D(size=(2, 1), name='upsample2')(x)
    x = ConvTrBNActBlock(x, name='block5', filter_dim=filter_dim, **conv_block_args)

    x = Conv2DTranspose(filters=num_channels, kernel_size=filter_size, padding='same',
                        kernel_initializer=kernel_initializer,
                        kernel_regularizer=kernel_regularizer,
                        name='output-convtr')(x)
    _x_out = Activation(activation=output_act, name='output-act')(x)

    ae_model = Model(inputs=_x_in, outputs=_x_out, name='AutoencoderModel')
    return ae_model


