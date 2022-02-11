from tensorflow import keras
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow import keras
import tensorflow as tf
import neptune as neptune
from tcn import TCN, tcn_full_summary
from utils.util_models import Time2Vec
from informer.informer_predictor import InformerPredictor
from tensorflow.keras.layers import Layer


class dataGeneratorSinLevel(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, df_survey,
                 list_IDs,
                 batch_size=16,
                 dim=50000,
                 vocab_size=1000,
                 n_classes=2,
                 shuffle=False,
                 if_deepsup=False,
                 if_softlabel=False,
                 if_static=False):
        'Initialization'
        self.df_survey = df_survey
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.vocab_size = vocab_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.if_deepsup = if_deepsup
        self.if_softlabel = if_softlabel
        self.on_epoch_end()
        self.static_cols = ['specialty_Anesthesiology',
       'specialty_Medicine', 'specialty_Pediatrics', 'year_in_program_1',
       'year_in_program_2', 'year_in_program_3', 'sex_Female', 'sex_Male',
       'race_Asian', 'race_Black', 'race_Hispanic', 'race_Other', 'race_White',
       'marital_status_Domestic', 'marital_status_Married',
       'marital_status_Single', 'children_No', 'children_Yes']
        self.if_static = if_static

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index + 1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        len_actions = self.dim
        df_survey = self.df_survey
        X_action = np.zeros((self.batch_size, self.dim, 1), dtype=int)
        X_time = np.zeros((self.batch_size, self.dim, 2), dtype=float)
        y_bin = np.zeros((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            y_bin[i] = df_survey[df_survey['sample_ID'] == ID]['burnout_bin'].map(int).values[0]
            # Store sample
            df = pd.read_csv('data_processed/logs_survey/logs_' + ID + '.csv')
            X_action[i, -len(df):, 0] = df['action_ID'].map(int).values[:len_actions]
            X_time[i, -len(df):, 0] = df['interval'].values[:len_actions]  # max value = 10 min
            X_time[i, -len(df):, 1] = df['time_of_day'].values[:len_actions] / 24

        labels = keras.utils.to_categorical(y_bin, num_classes=self.n_classes)

        return np.concatenate([X_action, X_time], axis=2), labels


class dataGeneratorSinLevelVariantHorizon(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, df_survey,
                 list_IDs,
                 batch_size=16,
                 dim=50000,
                 vocab_size=1000,
                 n_classes=2,
                 shuffle=False,
                 if_deepsup=False,
                 if_softlabel=False,
                 if_static=False,
                 horizon=0):
        'Initialization'
        self.df_survey = df_survey
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.vocab_size = vocab_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.if_deepsup = if_deepsup
        self.if_softlabel = if_softlabel
        self.on_epoch_end()
        self.static_cols = ['specialty_Anesthesiology',
       'specialty_Medicine', 'specialty_Pediatrics', 'year_in_program_1',
       'year_in_program_2', 'year_in_program_3', 'sex_Female', 'sex_Male',
       'race_Asian', 'race_Black', 'race_Hispanic', 'race_Other', 'race_White',
       'marital_status_Domestic', 'marital_status_Married',
       'marital_status_Single', 'children_No', 'children_Yes']
        self.if_static = if_static
        self.horizon = horizon

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index + 1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        len_actions = self.dim
        df_survey = self.df_survey
        X_action = np.zeros((self.batch_size, self.dim, 1), dtype=int)
        X_time = np.zeros((self.batch_size, self.dim, 2), dtype=float)
        y_bin = np.zeros((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            y_bin[i] = df_survey[df_survey['sample_ID'] == ID]['burnout_bin'].map(int).values[0]
            # Store sample
            df = pd.read_csv('data_processed/logs_survey/logs_' + ID + '.csv')
            #########
            if self.horizon != 0:
                shift_list = df['shift'].unique()
                df = df[df['shift'].isin(shift_list[:-self.horizon])]
            #########
            if len(df) > 0:
                X_action[i, -len(df):, 0] = df['action_ID'].map(int).values[:len_actions]
                X_time[i, -len(df):, 0] = df['interval'].values[:len_actions]  # max value = 10 min
                X_time[i, -len(df):, 1] = df['time_of_day'].values[:len_actions] / 24

        labels = keras.utils.to_categorical(y_bin, num_classes=self.n_classes)

        return np.concatenate([X_action, X_time], axis=2), labels


class singleLevelModel:
    def __init__(self, max_n_actions, embedding_matrix, dropout_rate=0.3):
        self.max_n_actions = max_n_actions
        self.embedding_matrix = embedding_matrix
        self.dropout_rate = dropout_rate
        self.tcn_layer = TCN(nb_filters=64,
                             nb_stacks=4,
                             dilations=[3 ** i for i in range(7)],
                             kernel_size=7,
                             dropout_rate=self.dropout_rate,
                             return_sequences=False,
                             use_weight_norm=True)
        print('ResTCN receptive field size =', self.tcn_layer.receptive_field)

        self.tcn_decoder = TCN(nb_filters=64,
                             nb_stacks=4,
                             dilations=[3 ** i for i in range(7)],
                             kernel_size=7,
                             dropout_rate=self.dropout_rate,
                             return_sequences=False,
                             use_weight_norm=True)
        print('ResTCN receptive field size =', self.tcn_decoder.receptive_field)

        self.simple_tcn_layer = simpleTCN(
            nb_filters=64,
            kernel_size=7,
            dilations=[3 for i in range(9)],
            n_feat_in=self.embedding_matrix.shape[1] + 50 * 2,
            n_feat_out=self.embedding_matrix.shape[1] + 50 * 2,
            max_len=max_n_actions,
            dropout_rate=dropout_rate,
            if_layer_norm=False
        )
        print('Single TCN bottleneck width =', 3008 // (2 ** 9))

    def embedding_layers(self, x):
        # inside shift
        x_action = layers.Lambda(lambda x: x[:, :, 0])(x)
        x_interv = layers.Lambda(lambda x: x[:, :, 1:2])(x)
        x_time = layers.Lambda(lambda x: x[:, :, 2:3])(x)

        # get embeddings of each component
        x_action = layers.Embedding(self.embedding_matrix.shape[0],
                                    self.embedding_matrix.shape[1],
                                    # embeddings_initializer=keras.initializers.Constant(self.embedding_matrix),
                                    mask_zero=True,
                                    trainable=True)(x_action)
        x_interv = layers.TimeDistributed(layers.Dense(50, activation='tanh', use_bias=True))(x_interv)
        x_time = Time2Vec(50, periodic_activation='sin', name='time')(x_time)

        # concatenation
        x = layers.Concatenate(axis=2, name='concatenate')([x_action, x_interv, x_time])
        # x = layers.Add(name='addition')([x_action, x_interv, x_time])

        return x

    def restcn(self):
        # encoding
        input = layers.Input(shape=(self.max_n_actions, 3))
        embeddings = self.embedding_layers(input)
        encoded = self.tcn_layer(embeddings)  # shape = (None, 64)

        predict = layers.Dense(64, activation='relu')(encoded)
        predict = layers.Dense(2, activation='softmax', name='prediction')(predict)

        model = models.Model(input, predict, name='SingleTCN')

        model.compile(loss='categorical_crossentropy',
                             optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                             metrics=['accuracy', tf.keras.metrics.AUC()])
        model.summary()

        return model

    def simpletcn(self):
        # encoding
        input = layers.Input(shape=(self.max_n_actions, 3))
        embeddings = self.embedding_layers(input)
        encoded = self.simple_tcn_layer(embeddings)  # shape = (None, 97, 64)
        encoded = layers.Flatten()(encoded)

        predict = layers.Dense(128, activation='relu')(encoded)
        predict = layers.Dense(2, activation='softmax', name='prediction')(predict)

        model = models.Model(input, predict, name='SingleTCN')

        model.compile(loss='categorical_crossentropy',
                      optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      metrics=['accuracy', tf.keras.metrics.AUC()])
        model.summary()

        return model

    def gru(self):
        # encoding
        input = layers.Input(shape=(self.max_n_actions, 3))
        embeddings = self.embedding_layers(input)
        encoded = layers.Bidirectional(layers.GRU(128))(embeddings)  # shape = (None, 64)

        predict = layers.Dense(64, activation='relu')(encoded)
        predict = layers.Dense(2, activation='softmax', name='prediction')(predict)

        model = models.Model(input, predict, name='SingleTCN')

        model.compile(loss='categorical_crossentropy',
                             optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                             metrics=['accuracy', tf.keras.metrics.AUC()])
        model.summary()

        return model

    def fcn_block(self, x):
        conv1 = layers.Conv1D(128, 8, 1, padding='same')(x)
        conv1 = layers.BatchNormalization()(conv1)
        conv1 = layers.Activation('relu')(conv1)

        drop_out1 = layers.Dropout(0.2)(conv1)
        conv2 = layers.Conv1D(256, 5, 1, padding='same')(drop_out1)
        conv2 = layers.BatchNormalization()(conv2)
        conv2 = layers.Activation('relu')(conv2)

        drop_out2 = layers.Dropout(0.2)(conv2)
        conv3 = layers.Conv1D(128, 3, 1, padding='same')(drop_out2)
        conv3 = layers.BatchNormalization()(conv3)
        out = layers.Activation('relu')(conv3)

        return out

    def fcn(self):


        input = layers.Input(shape=(self.max_n_actions, 3))
        embeddings = self.embedding_layers(input)
        #    drop_out = Dropout(0.2)(x)
        block1 = self.fcn_block(embeddings)
        block2 = self.fcn_block(block1)

        full = layers.GlobalAvgPool1D()(block2)
        predict = layers.Dense(2, activation='softmax', name='prediction')(full)

        model = models.Model(input, predict, name='SingleFCN')

        model.compile(loss='categorical_crossentropy',
                      optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      metrics=['accuracy', tf.keras.metrics.AUC()])
        model.summary()

        return model


    def informer(self, batch_size=None):
        informer = InformerPredictor(enc_in=200,
                                     seq_len=self.max_n_actions,
                                     batch_size=batch_size)
        # encoding
        input = layers.Input(shape=(self.max_n_actions, 3))
        embeddings = self.embedding_layers(input)
        predict = informer(embeddings)  # shape = (None, 64)

        model = models.Model(input, predict, name='SingleInformer')

        model.compile(loss='categorical_crossentropy',
                      optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                      metrics=['accuracy', tf.keras.metrics.AUC()])
        model.summary()

        return model


class simpleTCN(Layer):
    def __init__(self, nb_filters,
                 kernel_size,
                 dilations,
                 n_feat_in,
                 n_feat_out,
                 max_len,
                 dropout_rate=0.3,
                 if_layer_norm=False,
                 activation='relu'):
        super(simpleTCN, self).__init__()
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.dilations = dilations
        self.n_feat_in = n_feat_in
        self.n_feat_out = n_feat_out
        self.max_len = max_len
        self.dropout_rate = dropout_rate
        self.if_layer_norm = if_layer_norm
        self.activation = activation
        self.n_layers = len(dilations)
        self.conv_layers_1 = []
        self.conv_layers_2 = []
        self.activations = []
        self.dropout_layers = []
        self.max_pools = []
        self.layer_norms = []

    def build(self, input_shape):
        for k in range(self.n_layers):
            self.conv_layers_1.append(layers.Conv1D(
                filters=self.nb_filters,
                kernel_size=self.kernel_size,
                dilation_rate=self.dilations[k],
                padding='causal',
                name='conv_{}'.format(k),
                kernel_initializer='he_normal'
            ))
            # self.conv_layers_2.append(layers.Conv1D(
            #     filters=self.nb_filters,
            #     kernel_size=self.kernel_size,
            #     dilation_rate=self.dilations[k],
            #     padding='causal',
            #     name='conv_{}'.format(k),
            #     kernel_initializer='he_normal'
            # ))
            if self.if_layer_norm:
                self.layer_norms.append(
                layers.LayerNormalization()
                )
            self.activations.append(
                layers.Activation('relu')
            )
            self.dropout_layers.append(
                layers.SpatialDropout1D(rate=self.dropout_rate, name='en_dropout_{}'.format(k))
            )
            self.max_pools.append(
                layers.MaxPool1D(pool_size=2, name='maxpool_{}'.format(k))
            )

    def call(self, inputs, *args, **kwargs):
        # ---- Encoder ----
        x = inputs
        for k in range(self.n_layers):
            x = self.conv_layers_1[k](x)
            # x = self.conv_layers_2[k](x)
            if self.if_layer_norm: x = self.layer_norms[k](x)
            x = self.activations[k](x)
            x = self.dropout_layers[k](x)
            x = self.max_pools[k](x)

        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] // (2 ** self.tcn_layers), self.nb_filters)



