import keras as kr
import numpy as np

from keras import layers
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import Model, Input, Sequential
import keras.activations as activations


class RNN:
    def __init__(self, RNNconfig):
        self.config = RNNconfig
        self.input_shape = (self.config['num_steps'], self.config['sensors'])
        self.input = Input(self.input_shape)


    def build_model(self, inception=True, res=True, maxpool = True, avgpool = False, batchnorm=True):
        self.i = 0
        pad = 'same'
        padp = 'same'

        c_act = self.config['c_act']
        r_act = self.config['r_act']
        rk_act = self.config['rk_act']

        r = kr.regularizers.l2(self.config['reg'])


        a = self.input
        # a = layers.BatchNormalization()(a)


        if inception:
            c0 = layers.Conv1D(self.config['filters'], kernel_size=4, strides=self.config['strides'], padding=pad,
                               activation=c_act)(a)
            c1 = layers.Conv1D(self.config['filters'], kernel_size=8, strides=self.config['strides'], padding=pad,
                               activation=c_act)(a)
            c2 = layers.Conv1D(self.config['filters'], kernel_size=32, strides=self.config['strides'], padding=pad,
                               activation=c_act)(a)

            c = layers.concatenate([c0, c1, c2])

            if maxpool:
                c = layers.MaxPooling1D(2, padding=padp)(c)
            elif avgpool:
                c = layers.AveragePooling1D(2, padding=padp)(c) 
            if batchnorm:
                c = layers.BatchNormalization()(c)
            c = layers.SpatialDropout1D(self.config['cnn_drop'])(c)



            c0 = layers.Conv1D(self.config['filters'], kernel_size=4, strides=self.config['strides'], padding=pad,
                               activation=c_act)(c)
            c1 = layers.Conv1D(self.config['filters'], kernel_size=8, strides=self.config['strides'], padding=pad,
                               activation=c_act)(c)
            c2 = layers.Conv1D(self.config['filters'], kernel_size=32, strides=self.config['strides'], padding=pad,
                               activation=c_act)(c)

            c = layers.concatenate([c0, c1, c2])
            if maxpool:
                c = layers.MaxPooling1D(2, padding=padp)(c)
            elif avgpool:
                c = layers.AveragePooling1D(2, padding=padp)(c) 
            if batchnorm:
                c = layers.BatchNormalization()(c)
            c = layers.SpatialDropout1D(self.config['cnn_drop'])(c)



            c0 = layers.Conv1D(self.config['filters'], kernel_size=4, strides=self.config['strides'], padding=pad,
                               activation=c_act)(c)
            c1 = layers.Conv1D(self.config['filters'], kernel_size=8, strides=self.config['strides'], padding=pad,
                               activation=c_act)(c)
            c2 = layers.Conv1D(self.config['filters'], kernel_size=32, strides=self.config['strides'], padding=pad,
                               activation=c_act)(c)

            c = layers.concatenate([c0, c1, c2])
            if maxpool:
                c = layers.MaxPooling1D(2, padding=padp)(c)
            elif avgpool:
                c = layers.AveragePooling1D(2, padding=padp)(c) 
            if batchnorm:
                c = layers.BatchNormalization()(c)
            c = layers.SpatialDropout1D(self.config['cnn_drop'])(c)


        else:
            c = layers.Conv1D(self.config['filters'], kernel_size=4, strides=self.config['strides'], padding=pad, activation=c_act)(self.input)
            if maxpool:
                c = layers.MaxPooling1D(2, padding=padp)(c)
            elif avgpool:
                c = layers.AveragePooling1D(2, padding=padp)(c) 
            if batchnorm:
                c = layers.BatchNormalization()(c)
            c = layers.SpatialDropout1D(self.config['cnn_drop'])(c)

            c = layers.Conv1D(self.config['filters'], kernel_size=4, strides=self.config['strides'], padding=pad, activation=c_act)(c)
            if maxpool:
                c = layers.MaxPooling1D(2, padding=padp)(c)
            elif avgpool:
                c = layers.AveragePooling1D(2, padding=padp)(c) 
            if batchnorm:
                c = layers.BatchNormalization()(c)
            c = layers.SpatialDropout1D(self.config['cnn_drop'])(c)

            c = layers.Conv1D(self.config['filters'], kernel_size=4, strides=self.config['strides'], padding=pad, activation=c_act)(c)
            if maxpool:
                c = layers.MaxPooling1D(2, padding=padp)(c)
            elif avgpool:
                c = layers.AveragePooling1D(2, padding=padp)(c) 
            if batchnorm:
                c = layers.BatchNormalization()(c)
            c = layers.SpatialDropout1D(self.config['cnn_drop'])(c)

        if res:
            g1 = layers.GRU(self.config['state_size'], return_sequences=True, activation=rk_act, recurrent_activation=r_act,
                            dropout=self.config['rec_drop'], recurrent_dropout=self.config['rec_drop'],
                            recurrent_regularizer=r, kernel_regularizer=r)(c)
            g2 = layers.GRU(self.config['state_size'], return_sequences=True,  activation=rk_act, recurrent_activation=r_act,
                            dropout=self.config['rec_drop'], recurrent_dropout=self.config['rec_drop'],
                            recurrent_regularizer=r, kernel_regularizer=r)(g1)
            g_concat1 = layers.concatenate([g1, g2])

            g3 = layers.GRU(self.config['state_size'], return_sequences=True, activation=rk_act, recurrent_activation=r_act,
                            dropout=self.config['rec_drop'], recurrent_dropout=self.config['rec_drop'],
                            recurrent_regularizer=r, kernel_regularizer=r)(g_concat1)
            g_concat2 = layers.concatenate([g1, g2, g3])

            g = layers.GRU(self.config['state_size'], return_sequences=False, activation=rk_act, recurrent_activation=r_act,
                           dropout=self.config['rec_drop'], recurrent_dropout=self.config['rec_drop'],
                            recurrent_regularizer=r, kernel_regularizer=r)(g_concat2)

        else:
            g = layers.GRU(self.config['state_size'], return_sequences=True, activation=rk_act, recurrent_activation=r_act,
                           dropout=self.config['rec_drop'], recurrent_dropout=self.config['rec_drop'],
                            recurrent_regularizer=r, kernel_regularizer=r)(c)

            g = layers.GRU(self.config['state_size'], return_sequences=True, activation=rk_act, recurrent_activation=r_act,
                           dropout=self.config['rec_drop'], recurrent_dropout=self.config['rec_drop'],
                            recurrent_regularizer=r, kernel_regularizer=r)(g)
            g = layers.GRU(self.config['state_size'], return_sequences=True, activation=rk_act, recurrent_activation=r_act,
                           dropout=self.config['rec_drop'], recurrent_dropout=self.config['rec_drop'],
                            recurrent_regularizer=r, kernel_regularizer=r)(g)

            g = layers.GRU(self.config['state_size'], return_sequences=False, activation=rk_act, recurrent_activation=r_act,
                           dropout=self.config['rec_drop'], recurrent_dropout=self.config['rec_drop'],
                            recurrent_regularizer=r, kernel_regularizer=r)(g)


        d = layers.Dense(self.config['output_size'])(g)
        out = layers.Softmax()(d)

        self.model = Model(self.input, out)
        print("{} initialized.".format(self.model.name))

    # -!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!- #
    def build(self, index=0):
        self.i = index

        # RNN1
        a = self.input
        a = layers.Bidirectional(layers.GRU(self.config['state_size'], return_sequences=True, activation=self.config['r_act'], dropout=self.config['rec_drop']))(a)

        # Dense 1
        a = layers.TimeDistributed(layers.Dense(self.config['fc_size']))(a)
        a = layers.BatchNormalization()(a)
        a = layers.Dropout(self.config['drop'])(a)
        a = layers.Activation('relu')(a)


        a = layers.Bidirectional(layers.GRU(self.config['state_size2'], return_sequences=True, activation=self.config['r_act'], dropout=self.config['rec_drop']))(a)
        a = layers.Bidirectional(layers.GRU(self.config['state_size3'], return_sequences=False, activation=self.config['r_act'], dropout=self.config['rec_drop']))(a)

        a = layers.Dense(self.config['fc_size2'])(a)
        # a = layers.BatchNormalization()(a)
        # a = layers.Dropout(self.config['drop'])(a)
        # a = layers.Activation('relu')(a)

        a = layers.Dense(4)(a)
        a = layers.Activation('softmax')(a)

        self.model = Model(self.input, a)


    # -!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!- #
    def build_simple(self):
        r = kr.regularizers.l2(self.config['reg'])

        self.i = 0
        a = self.input

        # a = layers.Dense(self.config['fc_size2'])(a)
        # a = layers.BatchNormalization()(a)
        # a = layers.Dropout(self.config['drop'])(a)
        # a = layers.Activation('relu')(a)



        a = layers.GRU(self.config['state_size'], return_sequences=True,
                       activation=self.config['rk_act'], recurrent_activation=self.config['r_act'],
                       dropout=self.config['rec_drop'], recurrent_dropout=self.config['rec_drop'],
                       kernel_regularizer=r, recurrent_regularizer=r)(a)
        a = layers.GRU(self.config['state_size2'], return_sequences=False,
                       activation=self.config['rk_act'], recurrent_activation=self.config['r_act'],
                       dropout=self.config['rec_drop'], recurrent_dropout=self.config['rec_drop'],
                       kernel_regularizer=r, recurrent_regularizer=r)(a)

        a = layers.Dense(self.config['fc_size'], kernel_initializer='he_normal', kernel_regularizer=r)(a)

        a = layers.BatchNormalization()(a)
        a = layers.Activation('relu')(a)
        a = layers.Dropout(self.config['drop'])(a)

        a = layers.Dense(4)(a)
        a = layers.Softmax()(a)
        self.model = Model(self.input, a)

    # -!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!- #

    def build_conv1d(self):
        r = kr.regularizers.l2(self.config['reg'])
        a = self.input

        c1 = layers.Convolution1D(125, 20, strides=5, padding='same', kernel_regularizer=r)(a)
        c2 = layers.Convolution1D(125, 25, strides=5, padding='same', kernel_regularizer=r)(a)
        c3 = layers.Convolution1D(125, 50, strides=5, padding='same', kernel_regularizer=r)(a)
        c4 = layers.Convolution1D(125, 75, strides=5, padding='same', kernel_regularizer=r)(a)
        a = layers.concatenate([c1, c2, c3, c4])
        a = layers.BatchNormalization()(a)
        a = layers.Dropout(self.config['cnn_drop'])(a)
        a = layers.Activation('relu')(a)

        c1 = layers.Convolution1D(125, 20, strides=5, padding='same', kernel_regularizer=r)(a)
        c2 = layers.Convolution1D(125, 25, strides=5, padding='same', kernel_regularizer=r)(a)
        c3 = layers.Convolution1D(125, 50, strides=5, padding='same', kernel_regularizer=r)(a)
        c4 = layers.Convolution1D(125, 75, strides=5, padding='same', kernel_regularizer=r)(a)
        a = layers.concatenate([c1, c2, c3, c4])
        a = layers.BatchNormalization()(a)
        a = layers.Dropout(self.config['cnn_drop'])(a)
        a = layers.Activation('relu')(a)

        c1 = layers.Convolution1D(125, 20, strides=5, padding='same', kernel_regularizer=r)(a)
        c2 = layers.Convolution1D(125, 25, strides=5, padding='same', kernel_regularizer=r)(a)
        c3 = layers.Convolution1D(125, 50, strides=5, padding='same', kernel_regularizer=r)(a)
        c4 = layers.Convolution1D(125, 75, strides=5, padding='same', kernel_regularizer=r)(a)
        a = layers.concatenate([c1, c2, c3, c4])
        a = layers.BatchNormalization()(a)
        a = layers.Dropout(self.config['cnn_drop'])(a)
        a = layers.Activation('relu')(a)

        c1 = layers.Convolution1D(125, 20, strides=5, padding='same', kernel_regularizer=r)(a)
        c2 = layers.Convolution1D(125, 25, strides=5, padding='same', kernel_regularizer=r)(a)
        c3 = layers.Convolution1D(125, 50, strides=5, padding='same', kernel_regularizer=r)(a)
        c4 = layers.Convolution1D(125, 75, strides=5, padding='same', kernel_regularizer=r)(a)
        a = layers.concatenate([c1, c2, c3, c4])
        a = layers.BatchNormalization()(a)
        a = layers.Dropout(self.config['cnn_drop'])(a)
        a = layers.Activation('relu')(a)

        # c1 = layers.Convolution1D(125, 10, strides=5, padding='same', kernel_regularizer=r)(a)
        # c2 = layers.Convolution1D(125, 25, strides=5, padding='same', kernel_regularizer=r)(a)
        # c3 = layers.Convolution1D(125, 50, strides=5, padding='same', kernel_regularizer=r)(a)
        # c4 = layers.Convolution1D(125, 75, strides=5, padding='same', kernel_regularizer=r)(a)
        # a = layers.concatenate([c1, c2, c3, c4])
        # a = layers.BatchNormalization()(a)


        # a = layers.GRU(self.config['state_size'], return_sequences=True, activation='elu',
        #                dropout=self.config['rec_drop'], kernel_regularizer=r, recurrent_regularizer=r)(a)
        # a = layers.GRU(self.config['state_size2'], return_sequences=False, activation='elu',
        #                dropout=self.config['rec_drop'], kernel_regularizer=r, recurrent_regularizer=r)(a)

        a = layers.Flatten()(a)

        a = layers.Dense(4, kernel_initializer='he_normal', kernel_regularizer=r)(a)
        a = layers.Activation('softmax')(a)
        self.model = Model(self.input, a)

    # -!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!- #

    def build_cnn(self):
        self.i = 0
        self.input_shape = (self.config['sensors'], self.config['num_steps'], 1)
        self.input = Input(self.input_shape)

        self.model = Sequential([layers.InputLayer(self.input_shape),
                                layers.Conv2D(filters=20, kernel_size=(1, 25), activation=None),
                                layers.Conv2DTranspose(filters=20, kernel_size=(self.config['sensors'], 20)),
                                layers.AveragePooling2D((75, 1), strides=(15, 1), padding='same'),
                                layers.Flatten(),
                                layers.Dense(4),
                                layers.Activation('softmax')])




    def train(self, X, y, Xv=None, yv=None, verbose=1):
        print("Training {}".format(self.model.name))

        if Xv is None or yv is None:
            self.model.compile(loss=kr.losses.categorical_crossentropy,
                               optimizer=kr.optimizers.Adam(self.config['learning_rate']), metrics=['acc'])
            history = self.model.fit(x=X, y=y, batch_size=self.config['batch_size'],
                                     epochs=self.config['epochs'], verbose=verbose, validation_split=0.2, shuffle=True)

        else:
            self.model.compile(loss=kr.losses.categorical_crossentropy, optimizer=kr.optimizers.Adam(self.config['learning_rate']), metrics=['acc'])
            history = self.model.fit(x=X, y=y ,batch_size=self.config['batch_size'],
                                     epochs=self.config['epochs'], verbose=verbose,
                                    validation_data=(Xv, yv), shuffle=True)
        return history


    def predict_score(self, X):
        pred = self.model.predict(X)
        pred = np.argmax(pred, axis=1)  #TODO axis

        return pred


    def eval_error(self, X, y):
        pred = self.predict_score(X)

        return np.average(np.not_equal(np.argmax(y, axis=1), pred))


    def eval_acc(self, X, y):
        pred = self.predict_score(X)
        return np.average(np.equal(np.argmax(y, axis=1), pred))


