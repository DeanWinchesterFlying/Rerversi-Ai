from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras.losses import mean_squared_error
from keras.regularizers import l2
from keras.layers.merge import Add
import keras.backend as backend
import json
import os
from top.util.config import Config


class ReversiModel:
    def __init__(self, config: Config):
        self.config = config
        self.model = None

    def bulid(self):  # None => batch_size
        x_in = image_in = Input((2, 8, 8))  # (None, 2, 8, 8)
        conv1 = Conv2D(filters=self.config.model.cnn_filter_num,
                       kernel_size=self.config.model.cnn_filter_size,
                       strides=(1, 1), padding='same', data_format='channels_first',
                       kernel_regularizer=l2(self.config.model.l2_reg))(x_in)  # (None, 256, 8, 8)
        norm_conv1 = BatchNormalization(axis=-1)(conv1)  # (None, 256, 8, 8)
        acti_conv1 = Activation('relu')(norm_conv1)  # (None, 256, 8, 8)

        x = acti_conv1
        for _ in range(self.config.model.res_layer_num):  # residual block [x, h(x)]
            x = self._build_residual_block(x)
        res_out = x  # (None, 256, 8, 8)

        # value network
        x = Conv2D(filters=1, kernel_size=1, data_format="channels_first",
                   kernel_regularizer=l2(self.config.model.l2_reg))(x)  # (None, 1, 8, 8)
        x = BatchNormalization(axis=1)(x)  # (None, 1, 8, 8)
        x = Activation('relu')(x)  # (None, 1, 8, 8)
        x = Flatten()(x)  # (None, 64)
        value = Dense(units=1, kernel_regularizer=l2(self.config.model.l2_reg),
                      activation='tanh', name='value_out')(x)

        # policy network
        x = Conv2D(filters=2, kernel_size=1, data_format="channels_first",
                   kernel_regularizer=l2(self.config.model.l2_reg))(res_out)  # (None, 2, 8, 8)
        x = BatchNormalization(axis=1)(x)  # (None, 2, 8, 8)
        x = Activation('relu')(x)  # (None, 2, 8, 8)
        x = Flatten()(x)  # (None, 128)
        policy = Dense(units=64, kernel_regularizer=l2(self.config.model.l2_reg),
                       activation='softmax', name='policy_out')(x)
        self.model = Model(image_in, [policy, value], name='reversi_model')

    def predict(self, x):  # return policy, value
        assert x.ndim in (3, 4)
        assert x.shape == (2, 8, 8) or x.shape[1:] == (2, 8, 8)
        ndim = x.ndim
        if ndim == 3:
            x = x.reshape(1, 2, 8, 8)

        policy, value = self.model.predict_on_batch(x)
        if ndim == 3:  # batch size = 1
            return policy[0], value[0]
        return policy, value

    def _build_residual_block(self, x):
        x_in = x

        x = Conv2D(filters=self.config.model.cnn_filter_num,
                   kernel_size=self.config.model.cnn_filter_size,
                   strides=(1, 1), padding='same', data_format='channels_first',
                   kernel_regularizer=l2(self.config.model.l2_reg))(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)
        x = Conv2D(filters=self.config.model.cnn_filter_num,
                   kernel_size=self.config.model.cnn_filter_size,
                   strides=(1, 1), padding='same', data_format='channels_first',
                   kernel_regularizer=l2(self.config.model.l2_reg))(x)
        x = BatchNormalization(axis=-1)(x)
        x = Add()([x_in, x])
        x = Activation("relu")(x)
        return x

    def load(self, config_path, model_path):
        if os.path.exists(config_path) and os.path.exists(model_path):
            with open(config_path, 'r') as fp:
                self.model = Model.from_config(json.load(fp))
                self.model.load_weights(model_path)
            return True
        return False

    def save(self, config_path, model_path):
        with open(config_path, 'w') as fp:
            json.dump(self.model.get_config(), fp)
            self.model.save_weights(model_path)

    @staticmethod
    def value_loss(y_true, y_pred):  # square
        return mean_squared_error(y_true=y_true, y_pred=y_pred)

    @staticmethod
    def policy_loss(y_true, y_pred):  # log
        return backend.mean(-y_true * backend.log(y_pred + backend.epsilon()), axis=1)


if __name__ == "__main__":
    model = ReversiModel()
    model.bulid()