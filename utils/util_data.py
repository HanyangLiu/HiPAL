from tensorflow import keras
import numpy as np
import tensorflow as tf
import pandas as pd


class lowerDataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self,
                 list_IDs,  # here list_IDs should be from column 'sample_shift_ID'
                 batch_size=16,
                 dim=3000,
                 vocab_size=1000,
                 n_classes=2,
                 shuffle=False):
        'Initialization'
        self.len_shift = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.vocab_size = vocab_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

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
        X_action = np.zeros((self.batch_size, self.len_shift, 1), dtype=int)
        X_time = np.zeros((self.batch_size, self.len_shift, 2), dtype=float)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            df = pd.read_csv('data_processed/logs_shift/logs_' + ID + '.csv')
            X_action[i, -len(df):, 0] = df['action_ID'].map(int).values[:self.len_shift]
            X_time[i, -len(df):, 0] = df['interval'].values[:self.len_shift]
            X_time[i, -len(df):, 1] = df['time_of_day'].values[:self.len_shift] / 24

        return np.concatenate([X_action, X_time], axis=2), [X_action, X_time[:, :, 0], X_time[:, :, 1]]


class higherDataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, df_survey,
                 list_IDs,
                 batch_size=16,
                 dim=(30, 3000),
                 vocab_size=1000,
                 n_classes=2,
                 shuffle=False,
                 if_deepsup=False,
                 if_taildrop=False,
                 max_tail=5,
                 window=0
                 ):
        'Initialization'
        self.df_survey = df_survey
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.vocab_size = vocab_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.if_deepsup = if_deepsup
        self.if_taildrop = if_taildrop
        self.max_tail = max_tail
        self.window = window
        self.on_epoch_end()

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
        n_shifts = self.dim[0]
        len_shift = self.dim[1]
        df_survey = self.df_survey

        X_action = np.zeros((self.batch_size, *self.dim, 1), dtype=int)
        X_time = np.zeros((self.batch_size, *self.dim, 2), dtype=float)
        X_shift_time = np.zeros((self.batch_size, n_shifts, 2), dtype=float)
        y_bin = np.zeros((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            y_bin[i] = df_survey[df_survey['sample_ID'] == ID]['burnout_bin'].map(int).values[0]

            # read from files
            df = pd.read_csv('data_processed/logs_survey/logs_' + ID + '.csv')
            shift_IDs = df['shift'].unique()[-n_shifts:]
            shift_IDs = shift_IDs[-self.window:]

            if self.if_taildrop:
                distr = (np.linspace(1, 0, self.max_tail)) ** 2
                tail_len = np.random.choice(range(0, self.max_tail), p=distr / sum(distr))
                shift_IDs = shift_IDs[: -tail_len] if tail_len > 0 else shift_IDs

            for j, shift_ID in enumerate(shift_IDs):
                loc = n_shifts - len(shift_IDs) + j  # equivalent to zero-padding the head
                df_shift = df[df['shift'] == shift_ID]
                shift_interval = df_shift.iloc[0]['interval']
                df_shift.loc[df_shift.iloc[0:1].index, 'interval'] = 0

                # in-shift input
                X_action[i, loc, -len(df_shift):, 0] = df_shift['action_ID'].map(int).values[:len_shift]
                X_time[i, loc, -len(df_shift):, 0] = df_shift['interval'].values[:len_shift] # max value = 10 min
                X_time[i, loc, -len(df_shift):, 1] = df_shift['time_of_day'].values[:len_shift] / 24

                # out-shift input
                X_shift_time[i, loc, 0] = shift_interval
                X_shift_time[i, loc, 1] = df_shift.iloc[0]['days_to_survey'] / 30

        labels = keras.utils.to_categorical(y_bin, num_classes=self.n_classes)
        labels_rep = np.repeat(labels[:, np.newaxis, :], n_shifts, axis=1)

        inputs = [np.concatenate([X_action, X_time], axis=3), X_shift_time]
        outputs = [labels, labels_rep] if self.if_deepsup else labels

        return inputs, outputs
