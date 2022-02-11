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


class tcnEncoder(Layer):
    def __init__(self, nb_filters,
                 kernel_size,
                 dilations,
                 n_feat_in,
                 n_feat_out,
                 max_len,
                 dropout_rate=0.3,
                 if_layer_norm=False,
                 activation='relu'):
        super(tcnEncoder, self).__init__()
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
        self.conv_layers = []
        self.activations = []
        self.dropout_layers = []
        self.max_pools = []
        self.layer_norms = []

    def build(self, input_shape):
        for k in range(self.n_layers):
            self.conv_layers.append(layers.Conv1D(
                filters=self.nb_filters,
                kernel_size=self.kernel_size,
                dilation_rate=self.dilations[k],
                padding='causal',
                name='conv_{}'.format(k),
                kernel_initializer='he_normal'
            ))
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
            x = self.conv_layers[k](x)
            if self.if_layer_norm: x = self.layer_norms[k](x)
            x = self.activations[k](x)
            x = self.dropout_layers[k](x)
            x = self.max_pools[k](x)

        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] // (2 ** len(self.dilations)), self.nb_filters)


class tcnDecoder(Layer):
    def __init__(self, nb_filters,
                 kernel_size,
                 dilations,
                 n_feat_in,
                 n_feat_out,
                 max_len,
                 dropout_rate=0.3,
                 if_layer_norm=False,
                 activation='relu'):
        super(tcnDecoder, self).__init__()
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
        self.conv_layers = []
        self.activations = []
        self.dropout_layers = []
        self.up_samplings = []
        self.layer_norms = []

    def build(self, input_shape):
        for k in range(self.n_layers):
            self.conv_layers.append(layers.Convolution1DTranspose(
                filters=self.nb_filters,
                kernel_size=self.kernel_size,
                dilation_rate=self.dilations[self.n_layers - k - 1],
                padding='same',
                name='convtrans_{}'.format(k),
                kernel_initializer='he_normal',
            ))
            if self.if_layer_norm:
                self.layer_norms.append(
                layers.LayerNormalization()
                )
            self.activations.append(
                layers.Activation('relu')
            )
            self.dropout_layers.append(
                layers.SpatialDropout1D(rate=self.dropout_rate, name='de_dropout_{}'.format(k))
            )
            self.up_samplings.append(
                layers.UpSampling1D(size=2, name='upsampling_{}'.format(k))
            )

    def call(self, inputs, *args, **kwargs):
        # ---- Encoder ----
        x = inputs
        for k in range(self.n_layers):
            x = self.up_samplings[k](x)
            x = self.conv_layers[k](x)
            if self.if_layer_norm: x = self.layer_norms[k](x)
            x = self.activations[k](x)
            x = self.dropout_layers[k](x)

        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] * (2 ** self.tcn_layers), self.n_feat_out)


class dataEmbedding(Layer):
    def __init__(self, embedding_matrix,
                 time_size=None,
                 agg_mode='concat',
                 emb_mode=None,
                 if_emb_pre=True):
        super(dataEmbedding, self).__init__()
        self.embedding_matrix = embedding_matrix
        self.vocab_size, self.action_size = embedding_matrix.shape[0], embedding_matrix.shape[1]
        self.time_size = time_size

        if agg_mode == 'concat':
            self.agg_layer = layers.Concatenate(axis=2, name='lower_agg')
        elif agg_mode == 'add':
            self.time_size = self.action_size
            self.agg_layer = layers.Add(name='lower_agg')
        else:
            ValueError("Please specify embedding aggregation mode!")

        if if_emb_pre:
            initializer = keras.initializers.Constant(self.embedding_matrix)
        else:
            initializer = None

        self.action_embedding = layers.Embedding(
            self.vocab_size,
            self.action_size,
            embeddings_initializer=initializer,
            mask_zero=True,
            trainable=True,
            name='action_embedding'
        )
        self.interv_embedding = layers.TimeDistributed(
            layers.Dense(self.time_size, activation='tanh', use_bias=True),
            name='interv_embedding'
        )
        self.time_embedding = time2vec(self.time_size, periodic_activation='sin', name='time_embedding')
        self.emb_mode = emb_mode


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
        if self.emb_mode == 'wo_interval':
            x = self.agg_layer([x_action, x_time])
        elif self.emb_mode == 'wo_periodicity':
            x = self.agg_layer([x_action, x_interv])
        else:
            x = self.agg_layer([x_action, x_interv, x_time])

        return x

    def compute_output_shape(self, input_shape):
        if self.emb_mode:
            return (input_shape[0], input_shape[1], self.action_size + self.time_size)
        else:
            return (input_shape[0], input_shape[1], self.action_size + 2 * self.time_size)


class nestedLowerEncoding(Layer):
    def __init__(self, embedding_matrix,
                 tcn_layers,
                 time_size,
                 max_shift_len,
                 batch_size,
                 dropout,
                 emb_mode=None,
                 agg_mode=None,
                 if_emb_pre=True):
        super(nestedLowerEncoding, self).__init__()
        self.batch_size = batch_size
        self.max_shift_len = max_shift_len
        self.tcn_layers = tcn_layers
        self.embedding_matrix = embedding_matrix
        self.time_size = time_size
        self.dropout = dropout
        self.emb_mode = emb_mode
        self.agg_mode = agg_mode
        self.if_emb_pre = if_emb_pre
        if self.emb_mode == 'wo_interval' or self.emb_mode == 'wo_periodicity':
            self.input_size = self.embedding_matrix.shape[1] + self.time_size
        else:
            self.input_size = self.embedding_matrix.shape[1] + 2 * self.time_size

    def build(self, input_shape):
        self.data_embedding = dataEmbedding(self.embedding_matrix,
                                            time_size=self.time_size,
                                            agg_mode=self.agg_mode,
                                            emb_mode=self.emb_mode,
                                            if_emb_pre=self.if_emb_pre)
        self.encoder = tcnEncoder(
            nb_filters=64,
            kernel_size=5,
            dilations=[3 for i in range(self.tcn_layers)],
            n_feat_in=self.input_size,
            n_feat_out=self.input_size,
            max_len=self.max_shift_len,
            dropout_rate=self.dropout,
            if_layer_norm=False
        )

    def call(self, inputs, *args, **kwargs):
        x = inputs
        x = self.data_embedding(x)
        x = self.encoder(x)

        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] // (2 ** self.tcn_layers), 64)


class lowerModel:
    def __init__(self, embedding_matrix,
                 tcn_layers,
                 time_size,
                 max_shift_len,
                 batch_size,
                 dropout,
                 nested_lower_encoder,
                 emb_mode=None):
        self.batch_size = batch_size
        self.max_shift_len = max_shift_len
        self.embedding_matrix = embedding_matrix
        self.tcn_layers = tcn_layers
        self.time_size = time_size
        self.dropout = dropout
        self.nested_lower_encoder = nested_lower_encoder
        self.emb_mode = emb_mode
        if self.emb_mode == 'wo_interval' or self.emb_mode == 'wo_periodicity':
            self.input_size = self.embedding_matrix.shape[1] + self.time_size
        else:
            self.input_size = self.embedding_matrix.shape[1] + 2 * self.time_size
        self.build()

    def build(self):
        self.lower_decoder = tcnDecoder(
            nb_filters=64,
            kernel_size=5,
            dilations=[3 for i in range(self.tcn_layers)],
            n_feat_in=self.input_size,
            n_feat_out=self.input_size,
            max_len=self.max_shift_len,
            dropout_rate=0.3,
            if_layer_norm=False
        )
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
            self.args.dropout,
            self.args.emb_mode,
            self.args.agg_mode,
            self.args.if_emb_pre,
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







