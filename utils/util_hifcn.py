from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
from utils.util_hitcn_att import multiHeadSelfAttentionBlock, attLayer
# from keras_self_attention import SeqSelfAttention


class time2vec(Layer):
    def __init__(self, kernel_size, periodic_activation='sin', name='default'):
        '''
        :param kernel_size:         The length of time vector representation.
        :param periodic_activation: The periodic activation, sine or cosine, or any future function.
        '''
        super(time2vec, self).__init__(
            trainable=True,
            name='Time2Vec_' + periodic_activation + '_' + name
        )

        self.k = kernel_size - 1
        self.p_activation = periodic_activation

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'K': self.k,
            'activation': self.p_activation
        })
        return config

    def build(self, input_shape):
        # While i = 0
        self.wb = self.add_weight(
            shape=(1, 1),
            initializer='uniform',
            trainable=True,
            name='wb'
        )
        self.bb = self.add_weight(
            shape=(1, 1),
            initializer='uniform',
            trainable=True,
            name='bb'
        )
        # Else needs to pass the periodic activation
        self.wa = self.add_weight(
            shape=(1, self.k),
            initializer='uniform',
            trainable=True,
            name='wa'
        )
        self.ba = self.add_weight(
            shape=(1, self.k),
            initializer='uniform',
            trainable=True,
            name='ba'
        )
        super(time2vec, self).build(input_shape)

    def call(self, inputs, **kwargs):
        '''
        :param inputs: A Tensor with shape (batch_size, feature_size, 1)
        :param kwargs:
        :return: A Tensor with shape (batch_size, feature_size, length of time vector representation + 1)
        '''
        bias = self.wb * inputs + self.bb
        if self.p_activation.startswith('sin'):
            wgts = K.sin(K.dot(inputs, self.wa) + self.ba)
        elif self.p_activation.startswith('cos'):
            wgts = K.cos(K.dot(inputs, self.wa) + self.ba)
        else:
            raise NotImplementedError('Neither sine or cosine periodic activation be selected.')
        return K.concatenate([bias, wgts], -1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.k + 1)


class FCN(Layer):
    def __init__(self,
                 max_len,
                 dropout_rate=0.3):
        super(FCN, self).__init__()
        self.max_len = max_len
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.conv_layer_1 = layers.Conv1D(128, 8, 1, padding='same')
        self.bn_1 = layers.BatchNormalization()
        self.av_1 = layers.Activation('relu')
        self.do_1 = layers.Dropout(self.dropout_rate)

        self.conv_layer_2 = layers.Conv1D(256, 5, 1, padding='same')
        self.bn_2 = layers.BatchNormalization()
        self.av_2 = layers.Activation('relu')
        self.do_2 = layers.Dropout(self.dropout_rate)

        self.conv_layer_3 = layers.Conv1D(128, 3, 1, padding='same')
        self.bn_3 = layers.BatchNormalization()
        self.av_3 = layers.Activation('relu')

    def call(self, inputs, *args, **kwargs):
        # ---- Encoder ----
        x = inputs
        conv1 = self.conv_layer_1(x)
        conv1 = self.bn_1(conv1)
        conv1 = self.av_1(conv1)
        drop_out1 = self.do_1(conv1)

        conv2 = self.conv_layer_2(drop_out1)
        conv2 = self.bn_2(conv2)
        conv2 = self.av_2(conv2)
        drop_out2 = self.do_2(conv2)

        conv3 = self.conv_layer_3(drop_out2)
        conv3 = self.bn_3(conv3)
        conv3 = self.av_3(conv3)

        return conv3

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.max_len, 128)


class dataEmbedding(Layer):
    def __init__(self, embedding_matrix,
                 time_size=None,
                 agg_mode='concat'):
        super(dataEmbedding, self).__init__()
        self.embedding_matrix = embedding_matrix
        self.vocab_size, self.action_size = embedding_matrix.shape[0], embedding_matrix.shape[1]
        self.time_size = time_size
        self.action_embedding = layers.Embedding(
            self.vocab_size,
            self.action_size,
            embeddings_initializer=keras.initializers.Constant(self.embedding_matrix),
            mask_zero=True,
            trainable=True,
            name='action_embedding'
        )
        self.interv_embedding = layers.TimeDistributed(
            layers.Dense(self.time_size, activation='tanh', use_bias=True),
            name='interv_embedding'
        )
        self.time_embedding = time2vec(self.time_size, periodic_activation='sin', name='time_embedding')

        if agg_mode == 'concat':
            self.agg_layer = layers.Concatenate(axis=2, name='lower_agg')
        elif agg_mode == 'add':
            self.time_size = self.vocab_size
            self.agg_layer = layers.Add(name='lower_agg')
        else:
            ValueError("Please specify embedding aggregation mode!")


    def call(self, inputs, *args, **kwargs):
        x = inputs

        # inside shift
        x_action = layers.Lambda(lambda x: x[:, :, 0])(x)
        x_interv = layers.Lambda(lambda x: x[:, :, 1:2])(x)
        x_time = layers.Lambda(lambda x: x[:, :, 2:3])(x)

        # get embeddings of each component
        x_action = self.action_embedding(x_action)
        x_interv = self.interv_embedding(x_interv)
        x_time = self.time_embedding(x_time)

        # concatenation
        x = self.agg_layer([x_action, x_interv, x_time])

        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.action_size + 2 * self.time_size)


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
        self.encoder = FCN(max_len=self.max_shift_len, dropout_rate=self.dropout)

    def call(self, inputs, *args, **kwargs):
        x = inputs
        x = self.data_embedding(x)
        x = self.encoder(x)

        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.max_shift_len, 128)


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
        self.pool_layer = layers.GlobalAvgPool1D()
        self.repeat_vector = layers.RepeatVector(self.max_shift_len)
        self.lower_decoder = FCN(max_len=self.max_shift_len, dropout_rate=self.dropout)
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
        x = self.pool_layer(x)
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
        self.pool_layer = layers.GlobalAvgPool1D()
        self.nested_base = layers.TimeDistributed(self.nested_lower_encoder, name='nested_base_model')
        self.nested_pool = layers.TimeDistributed(self.pool_layer, name='nested_pool')
        self.nested_dense = layers.TimeDistributed(
            layers.Dense(128, activation='relu'),
            name='nested_dense'
        )
        self.get_interv = layers.Lambda(lambda x: x[:, :, 0:1])
        self.get_time = layers.Lambda(lambda x: x[:, :, 1:2])
        self.higher_interv = layers.TimeDistributed(
            layers.Dense(32, activation='tanh', use_bias=True),
            name='higher_interv_emb'
        )
        self.higher_time = time2vec(32, periodic_activation='sin', name='higher_time_emb')
        self.higher_agg = layers.Concatenate(axis=2, name='higher_agg')
        self.higher_agg_2 = layers.Concatenate(axis=2, name='higher_agg_2')
        self.higher_seq_model = layers.Bidirectional(layers.LSTM(64, return_sequences=False), name='bi-lstm')
        self.last_softmax = layers.Dense(2, activation='softmax', name='prediction')
        self.all_softmax = layers.TimeDistributed(layers.Dense(2, activation='softmax'), name='target_rep')

        self.nested_dropout = layers.Dropout(self.dropout)
        self.interv_dropout = layers.Dropout(self.dropout)

    def call(self):
        input_1 = layers.Input((self.max_n_shift, self.max_shift_len, 3))
        input_2 = layers.Input((self.max_n_shift, 2))

        x_1 = self.nested_base(input_1)
        x_1 = self.nested_pool(x_1)
        x_1 = self.nested_dense(x_1)
        x_1 = self.nested_dropout(x_1)

        input_interv = self.get_interv(input_2)
        input_day = self.get_time(input_2)
        x_2 = self.higher_interv(input_interv)
        x_2 = self.interv_dropout(x_2)
        x_3 = self.higher_time(input_day)

        x = self.higher_agg([x_1, x_2, x_3])
        lstm_out = self.higher_seq_model(x)
        last_pred = self.last_softmax(lstm_out)
        all_preds = self.all_softmax(x)
        model = tf.keras.models.Model([input_1, input_2], last_pred, name='higher_model')

        if self.if_deepsup:
            model = tf.keras.models.Model([input_1, input_2], [last_pred, all_preds], name='higher_model')

        return model


class buildHierTCN:
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







