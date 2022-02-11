from tensorflow import keras
import numpy as np
import tensorflow as tf
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
import random


class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, df_survey,
                 list_IDs,
                 batch_size=16,
                 dim=(30, 3000),
                 vocab_size=1000,
                 n_classes=2,
                 shuffle=False,
                 if_deepsup=False,
                 if_softlabel=False):
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
        X_static = np.zeros((self.batch_size, 18), dtype=float)

        y_bin = np.zeros((self.batch_size), dtype=int)
        y_soft = np.zeros((self.batch_size, 2), dtype=float)
        y_reg = np.zeros((self.batch_size, 1), dtype=float)
        y_score = np.zeros((self.batch_size, n_shifts, 1), dtype=float)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            X_static[i, :] = df_survey[df_survey['sample_ID'] == ID][self.static_cols]
            y_bin[i] = df_survey[df_survey['sample_ID'] == ID]['burnout_bin'].map(int).values[0]
            y_soft[i, 0] = df_survey[df_survey['sample_ID'] == ID]['burnout_soft'].values[0]
            y_soft[i, 1] = 1 - y_soft[i, 0]
            y_reg[i, 0] = df_survey[df_survey['sample_ID'] == ID]['burnout'].values[0]
            y_score[i, :, 0] = df_survey[df_survey['sample_ID'] == ID]['burnout'].values[0]
            # Store sample
            df = pd.read_csv('data_processed/logs_survey/logs_' + ID + '.csv')
            shift_IDs = df['shift'].unique()[-n_shifts:]
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

        labels = keras.utils.to_categorical(y_bin, num_classes=self.n_classes) if not self.if_softlabel else y_soft
        labels_rep = np.repeat(labels[:, np.newaxis, :], n_shifts, axis=1)

        if self.if_deepsup:
            return [np.concatenate([X_action, X_time], axis=3), X_shift_time], [labels, labels_rep]
        else:
            return [np.concatenate([X_action, X_time], axis=3), X_shift_time], labels


class DataGeneratorFixLen(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, df_logs, df_survey,
                 list_IDs,
                 batch_size=16,
                 dim=(20, 2000),
                 vocab_size=1000,
                 n_classes=2,
                 shuffle=False,
                 if_deepsup=False):
        'Initialization'
        self.df_logs = df_logs
        self.df_survey = df_survey
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.vocab_size = vocab_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.if_deepsup = if_deepsup
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
        df_logs = self.df_logs
        df_survey = self.df_survey
        len_max = self.dim[0] * self.dim[1]
        X_action = np.zeros((self.batch_size, len_max, 1), dtype=int)
        X_time = np.zeros((self.batch_size, len_max, 2), dtype=float)

        y_bin = np.zeros((self.batch_size), dtype=int)
        y_reg = np.zeros((self.batch_size, 1), dtype=float)
        y_score = np.zeros((self.batch_size, n_shifts, 1), dtype=float)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            df = df_logs[df_logs['sample_ID'] == ID]
            X_action[i, -min(len(df), len_max):, 0] = df['action_ID'][:len_max]
            X_time[i, -min(len(df), len_max):, 0] = df['interval'][:len_max]
            X_time[i, -min(len(df), len_max):, 1] = df['time_of_day'][:len_max] / 24

            y_bin[i] = df_survey[df_survey['sample_ID'] == ID]['burnout_bin'].map(int).values[0]
            y_reg[i, 0] = df_survey[df_survey['sample_ID'] == ID]['burnout'].values[0]
            y_score[i, :, 0] = df_survey[df_survey['sample_ID'] == ID]['burnout'].values[0]

        X_action = X_action.reshape((self.batch_size, *self.dim, 1))
        X_time = X_time.reshape((self.batch_size, *self.dim, 2))

        labels = keras.utils.to_categorical(y_bin, num_classes=self.n_classes)
        labels_rep = np.repeat(labels[:, np.newaxis, :], n_shifts, axis=1)

        if self.if_deepsup:
            return np.concatenate([X_action, X_time], axis=3), [labels, labels_rep]
        else:
            return np.concatenate([X_action, X_time], axis=3), labels


class DataGeneratorInner(tf.keras.utils.Sequence):
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


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


class DataGeneratorOuter(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, df_survey,
                 list_IDs,
                 batch_size=16,
                 dim=(30, 3000),
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
        n_shifts = self.dim[0]
        len_shift = self.dim[1]
        df_survey = self.df_survey
        X_action = np.zeros((self.batch_size, *self.dim, 1), dtype=int)
        X_time = np.zeros((self.batch_size, *self.dim, 2), dtype=float)
        X_static = np.zeros((self.batch_size, 18), dtype=float)
        X_shift_time = np.zeros((self.batch_size, n_shifts, 2), dtype=float)

        y_bin = np.zeros((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            X_static[i, :] = df_survey[df_survey['sample_ID'] == ID][self.static_cols]
            y_bin[i] = df_survey[df_survey['sample_ID'] == ID]['burnout_bin'].map(int).values[0]
            # Store sample
            df = pd.read_csv('data_processed/logs_survey/logs_' + ID + '.csv')
            shift_IDs = df['shift'].unique()[-n_shifts:]
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

        if self.if_deepsup:
            if self.if_static:
                return [np.concatenate([X_action, X_time], axis=3), X_shift_time, X_static], [labels, labels_rep]
            else:
                return [np.concatenate([X_action, X_time], axis=3), X_shift_time], [labels, labels_rep]
        else:
            if self.if_static:
                return [np.concatenate([X_action, X_time], axis=3), X_shift_time, X_static], labels
            else:
                return [np.concatenate([X_action, X_time], axis=3), X_shift_time], labels


class DataGeneratorVariantHorizon(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, df_survey,
                 list_IDs,
                 batch_size=16,
                 dim=(30, 3000),
                 vocab_size=1000,
                 n_classes=2,
                 shuffle=False,
                 if_deepsup=False,
                 if_softlabel=False,
                 horizon=0,
                 if_rand_horizon=True,
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
        self.horizon = horizon
        self.if_rand_horizon = if_rand_horizon
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
        n_shifts = self.dim[0]
        len_shift = self.dim[1]
        df_survey = self.df_survey
        X_action = np.zeros((self.batch_size, *self.dim, 1), dtype=int)
        X_time = np.zeros((self.batch_size, *self.dim, 2), dtype=float)
        X_static = np.zeros((self.batch_size, 18), dtype=float)
        X_shift_time = np.zeros((self.batch_size, n_shifts, 2), dtype=float)

        y_bin = np.zeros((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            X_static[i, :] = df_survey[df_survey['sample_ID'] == ID][self.static_cols]
            y_bin[i] = df_survey[df_survey['sample_ID'] == ID]['burnout_bin'].map(int).values[0]
            # Store sample
            df = pd.read_csv('data_processed/logs_survey/logs_' + ID + '.csv')
            shift_IDs = df['shift'].unique()[-n_shifts:]
            #########
            if self.if_rand_horizon:
                distr = (np.linspace(1, 0, self.horizon)) ** 2
                horizon = np.random.choice(range(0, self.horizon), p=distr / sum(distr))
            else:
                horizon = self.horizon
            shift_IDs = shift_IDs[: -horizon] if horizon > 0 else shift_IDs
            #########
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

        if self.if_deepsup:
            if self.if_static:
                return [np.concatenate([X_action, X_time], axis=3), X_shift_time, X_static], [labels, labels_rep]
            else:
                return [np.concatenate([X_action, X_time], axis=3), X_shift_time], [labels, labels_rep]
        else:
            if self.if_static:
                return [np.concatenate([X_action, X_time], axis=3), X_shift_time, X_static], labels
            else:
                return [np.concatenate([X_action, X_time], axis=3), X_shift_time], labels


class DataGeneratorVariantWindow(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, df_survey,
                 list_IDs,
                 batch_size=16,
                 dim=(30, 3000),
                 vocab_size=1000,
                 n_classes=2,
                 shuffle=False,
                 if_deepsup=False,
                 if_softlabel=False,
                 window=0,
                 if_rand_horizon=True):
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
        self.window = window
        self.if_rand_horizon = if_rand_horizon

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
        X_static = np.zeros((self.batch_size, 18), dtype=float)
        X_shift_time = np.zeros((self.batch_size, n_shifts, 2), dtype=float)

        y_bin = np.zeros((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            X_static[i, :] = df_survey[df_survey['sample_ID'] == ID][self.static_cols]
            y_bin[i] = df_survey[df_survey['sample_ID'] == ID]['burnout_bin'].map(int).values[0]
            # Store sample
            df = pd.read_csv('data_processed/logs_survey/logs_' + ID + '.csv')
            shift_IDs = df['shift'].unique()[-n_shifts:]
            #########
            window = self.window
            shift_IDs = shift_IDs[-window:]
            #########
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

        if self.if_deepsup:
            return [np.concatenate([X_action, X_time], axis=3), X_shift_time], [labels, labels_rep]
        else:
            return [np.concatenate([X_action, X_time], axis=3), X_shift_time], labels

