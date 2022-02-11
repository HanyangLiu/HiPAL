from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
from utils.util_hitcn import time2vec, dataEmbedding


class nestedLowerEncoding(Layer):
    def __init__(self, embedding_matrix,
                 tcn_layers,
                 time_size,
                 max_shift_len,
                 batch_size,
                 dropout):
        super(nestedLowerEncoding, self).__init__()
        self.batch_size = batch_size
        self.max_shift_len = max_shift_len
        self.tcn_layers = tcn_layers
        self.embedding_matrix = embedding_matrix
        self.time_size = time_size
        self.dropout = dropout

    def build(self, input_shape):
        self.data_embedding = dataEmbedding(self.embedding_matrix, time_size=self.time_size, agg_mode='concat')
        self.encoder = layers.LSTM(128)

    def call(self, inputs, *args, **kwargs):
        x = inputs
        x = self.data_embedding(x)
        x = self.encoder(x)

        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 128)


class lowerModel:
    def __init__(self, embedding_matrix,
                 tcn_layers,
                 time_size,
                 max_shift_len,
                 batch_size,
                 dropout,
                 nested_lower_encoder):
        self.batch_size = batch_size
        self.max_shift_len = max_shift_len
        self.embedding_matrix = embedding_matrix
        self.tcn_layers = tcn_layers
        self.time_size = time_size
        self.dropout = dropout
        self.nested_lower_encoder = nested_lower_encoder
        self.build()

    def build(self):
        self.lower_decoder = layers.Bidirectional(layers.GRU(128, return_sequences=True))
        self.repeat_vector = layers.RepeatVector(self.max_shift_len)
        self.action_extract = layers.TimeDistributed(
            layers.Dense(self.embedding_matrix.shape[1], activation='relu'),
            name='action_extract'
        )
        self.action_recon = layers.TimeDistributed(
            layers.Dense(self.embedding_matrix.shape[0],
                         activation='softmax',
                         kernel_initializer=keras.initializers.Constant(self.embedding_matrix.transpose())),
            name='action_recon',
            trainable=True)
        self.interv_extract = layers.TimeDistributed(
            layers.Dense(self.time_size, activation='relu'),
            name='interv_extract'
        )
        self.interv_recon = layers.TimeDistributed(
            layers.Dense(1, activation='linear'),
            name='interv_recon'
        )
        self.time_extract = layers.TimeDistributed(
            layers.Dense(self.time_size, activation='relu'),
            name='time_extract'
        )
        self.time_recon = layers.TimeDistributed(
            layers.Dense(1, activation='linear'),
            name='time_recon'
        )

    def call(self):
        inputs = layers.Input(shape=(self.max_shift_len, 3))
        x = inputs

        x = self.nested_lower_encoder(x)
        x = self.repeat_vector(x)
        x = self.lower_decoder(x)

        action = self.action_extract(x)
        action = self.action_recon(action)

        interv = self.interv_extract(x)
        interv = self.interv_recon(interv)

        time = self. time_extract(x)
        time = self.time_recon(time)

        model = tf.keras.models.Model(inputs, [action, interv, time], name='lower_model')

        return model


class higherModel:
    def __init__(self, embedding_matrix,
                 tcn_layers,
                 time_size,
                 max_shift_len,
                 max_n_shift,
                 batch_size,
                 dropout=0.3,
                 if_deepsup=False,
                 nested_lower_encoder=None):

        self.max_n_shift = max_n_shift
        self.max_shift_len = max_shift_len
        self.batch_size = batch_size
        self.action_size = embedding_matrix.shape[1]
        self.time_size = time_size
        self.dropout = dropout
        self.if_deepsup = if_deepsup
        self.tcn_layers = tcn_layers
        self.nested_lower_encoder = nested_lower_encoder
        self.embedding_matrix = embedding_matrix
        self.build()

    def build(self):
        self.pool_layer = layers.Flatten()
        self.nested_base = layers.TimeDistributed(self.nested_lower_encoder, name='nested_base_model')
        self.get_interv = layers.Lambda(lambda x: x[:, :, 0:1])
        self.get_time = layers.Lambda(lambda x: x[:, :, 1:2])
        self.higher_interv = layers.TimeDistributed(
            layers.Dense(32, activation='tanh', use_bias=True),
            name='higher_interv_emb'
        )
        self.higher_time = time2vec(32, periodic_activation='sin', name='higher_time_emb')
        self.higher_agg = layers.Concatenate(axis=2, name='higher_agg')
        self.higher_seq_model = layers.Bidirectional(layers.LSTM(64, return_sequences=False), name='bi-lstm')
        self.get_last_output = layers.Dense(2, activation='softmax', name='prediction')
        self.target_replication = layers.TimeDistributed(layers.Dense(2, activation='softmax'), name='target_rep')

        self.nested_dropout = layers.Dropout(self.dropout)
        self.interv_dropout = layers.Dropout(self.dropout)

    def call(self):
        input_1 = layers.Input((self.max_n_shift, self.max_shift_len, 3))
        input_2 = layers.Input((self.max_n_shift, 2))

        x_1 = self.nested_base(input_1)

        input_interv = self.get_interv(input_2)
        input_day = self.get_time(input_2)
        x_2 = self.higher_interv(input_interv)
        x_2 = self.interv_dropout(x_2)
        x_3 = self.higher_time(input_day)

        x = self.higher_agg([x_1, x_2, x_3])
        last_state = self.higher_seq_model(x)
        last_pred = self.get_last_output(last_state)
        all_preds = self.target_replication(x)
        model = tf.keras.models.Model([input_1, input_2], last_pred, name='higher_model')

        if self.if_deepsup:
            model = tf.keras.models.Model([input_1, input_2], [last_pred, all_preds], name='higher_model')

        return model


class buildHierGRU:
    def __init__(self, args, embedding_matrix):
        self.args = args
        self.embedding_matrix = embedding_matrix
        self.nested_lower_encoder = nestedLowerEncoding(
            self.embedding_matrix,
            self.args.tcn_layers,
            self.args.time_size,
            self.args.max_shift_len,
            self.args.bs_lower * self.args.max_n_shift,
            self.args.dropout
        )
        self.lower_model = lowerModel(
            self.embedding_matrix,
            self.args.tcn_layers,
            self.args.time_size,
            self.args.max_shift_len,
            self.args.bs_lower,
            self.args.dropout,
            self.nested_lower_encoder
        ).call()
        self.higher_model = higherModel(
            self.embedding_matrix,
            self.args.tcn_layers,
            self.args.time_size,
            self.args.max_shift_len,
            self.args.max_n_shift,
            self.args.bs_higher,
            dropout=self.args.dropout,
            if_deepsup=self.args.if_deepsup,
            nested_lower_encoder=self.nested_lower_encoder
        ).call()

    def build_lower_model(self):
        lower_model = self.lower_model
        lower_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.args.lr),
            loss={'action_recon': 'sparse_categorical_crossentropy',
                  'interv_recon': 'mse',
                  'time_recon': 'mse'},
            metrics = {'action_recon': 'accuracy'},
            run_eagerly=True
        )

        return lower_model

    def build_higher_model(self):
        higher_model = self.higher_model
        if self.args.if_deepsup:
            higher_model.compile(
                loss={'prediction': 'categorical_crossentropy', 'target_rep': inner_ce_loss},
                loss_weights={'prediction': 1, 'target_rep': self.args.alpha},
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.args.lr),
                metrics={'prediction': ['accuracy', tf.keras.metrics.AUC()], 'target_rep': 'accuracy'}
            )
        else:
            higher_model.compile(
                loss='categorical_crossentropy',
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.args.lr),
                metrics=['accuracy', tf.keras.metrics.AUC()]
            )

        return higher_model


def inner_ce_loss(true, pred):
    losses = tf.keras.losses.binary_crossentropy(true, pred)

    return K.mean(losses, axis=-1)







