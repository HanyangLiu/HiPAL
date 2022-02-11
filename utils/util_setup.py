import neptune
from gensim.models import KeyedVectors
import pandas as pd
import numpy as np
from utils.util_data import lowerDataGenerator, higherDataGenerator
from archive.util_data_semi import dataGeneratorSemi
from utils.util_single_level import dataGeneratorSinLevel
from utils.util_semi_hitcn import semiDataGenerator
import os
import tensorflow as tf
import random
from sklearn.model_selection import train_test_split


def init_neptune(args):
    # neptune logs
    neptune.set_project('hanyang.liu/burnout')
    neptune.init('hanyang.liu/burnout')
    PARAMS = args.__dict__
    neptune.create_experiment(name=args.memo, params=PARAMS,
                              upload_source_files=['hitcn.py', 'utils/*'])


def set_random_seeds(SEED):
   os.environ['PYTHONHASHSEED']=str(SEED)
   tf.random.set_seed(SEED)
   np.random.seed(SEED)
   random.seed(SEED)


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def RandomizedGroupKFold(groups, n_splits, random_state=None):  # noqa: N802
    """
    Random analogous of sklearn.model_selection.GroupKFold.split.
    :return: list of (train, test) indices
    """
    groups = pd.Series(groups)
    ix = np.arange(len(groups))
    unique = np.unique(groups)
    np.random.RandomState(random_state).shuffle(unique)
    result = []
    for split in np.array_split(unique, n_splits):
        mask = groups.isin(split)
        train, test = ix[~mask], ix[mask]
        result.append((train, test))

    return result


def data_loader_inner(train_uids, val_uids):
    logs_file = 'data_processed/logs_all.csv'
    survey_file = 'data_processed/survey_processed.csv'

    # load files
    survey = pd.read_csv(survey_file)
    logs_all = pd.read_csv(logs_file); logs_all['USER_ID'] = logs_all['USER_ID'].astype(str)  # ensure USER_ID is in string type

    # shift as unit get IDs for training and testing set
    IDs_train = logs_all[logs_all['USER_ID'].isin(train_uids)]['sample_shift_ID'].unique().tolist()
    IDs_test = logs_all[logs_all['USER_ID'].isin(val_uids)]['sample_shift_ID'].unique().tolist()
    IDs_train, IDs_val = train_test_split(IDs_train, test_size=1/8, random_state=123)

    return logs_all, survey, (IDs_train, IDs_val, IDs_test)


def data_loader_outer(train_uids, val_uids):
    logs_file = 'data_processed/logs_all.csv'
    survey_file = 'data_processed/survey_processed.csv'
    static_file = 'data_processed/static_feats.csv'

    # load files
    df_survey = pd.read_csv(survey_file)
    static = pd.read_csv(static_file)
    logs_all = pd.read_csv(logs_file); logs_all['USER_ID'] = logs_all['USER_ID'].astype(str)  # ensure USER_ID is in string type

    # add static features to df_survey table
    df_survey = pd.concat([df_survey, static.iloc[:, 5:23]], axis=1)

    # exclude invalid surveys
    df_survey = df_survey[df_survey['burnout_bin'].notna()]
    logs_all = logs_all[logs_all['sample_ID'].isin(df_survey['sample_ID'].values)]

    # exclude surveys with no actions
    action_count = logs_all.groupby('sample_ID').count()['action']
    logs_all = logs_all[logs_all['sample_ID'].isin(action_count[action_count >= 0].index)]
    df_survey = df_survey[df_survey['sample_ID'].isin(logs_all['sample_ID'].unique())]

    # get sample_ID for training and testing set
    IDs_train = df_survey[df_survey['USER_ID'].isin(train_uids)]['sample_ID'].values.tolist()
    IDs_test = df_survey[df_survey['USER_ID'].isin(val_uids)]['sample_ID'].values.tolist()
    IDs_train, IDs_val = train_test_split(IDs_train, test_size=1/8, random_state=123)

    return logs_all, df_survey, (IDs_train, IDs_val, IDs_test)


def data_loader_semi(train_uids, val_uids):
    logs_file = 'data_processed/logs_all.csv'
    survey_file = 'data_processed/survey_processed.csv'
    static_file = 'data_processed/static_feats.csv'

    # load files
    df_survey = pd.read_csv(survey_file)
    static = pd.read_csv(static_file)
    logs_all = pd.read_csv(logs_file); logs_all['USER_ID'] = logs_all['USER_ID'].astype(str)  # ensure USER_ID is in string type

    # add static features to df_survey table
    df_survey = pd.concat([df_survey, static.iloc[:, 5:23]], axis=1)

    # exclude invalid surveys
    df_survey = df_survey[df_survey['burnout_bin'].notna()]

    # exclude surveys with no actions
    action_count = logs_all.groupby('sample_ID').count()['action']
    logs_all = logs_all[logs_all['sample_ID'].isin(action_count[action_count >= 0].index)]
    df_survey = df_survey[df_survey['sample_ID'].isin(logs_all['sample_ID'].unique())]

    # get sample_ID for training and testing set
    IDs_test = df_survey[df_survey['USER_ID'].isin(val_uids)]['sample_ID'].values.tolist()
    IDs_train = logs_all['sample_ID'].unique().tolist()
    IDs_train, IDs_test = train_test_split(IDs_train, test_size=1/8, random_state=123)

    return logs_all, df_survey, (IDs_train, IDs_val, IDs_test)


def get_embeddings(dim_embedding):
    word_vector_file = 'data_processed/action_vectors_' + str(dim_embedding) + '.wv'
    wv = KeyedVectors.load(word_vector_file)
    vocab_size = len(wv.vocab) + 1
    embedding_matrix = np.zeros((vocab_size, dim_embedding))
    for action_ID in wv.vocab.keys():
        embedding_vector = wv.get_vector(action_ID)
        embedding_matrix[int(action_ID), :] = embedding_vector

    return embedding_matrix


def get_generators_lower(args, embedding_matrix, IDs):
    IDs_train, IDs_val, IDs_test = IDs

    train_generator = lowerDataGenerator(
        list(set(IDs_train + IDs_val + IDs_test)),
        batch_size=args.bs_lower,
        dim=args.max_shift_len,
        vocab_size=embedding_matrix.shape[0],
        shuffle=True
    )
    valid_generator = lowerDataGenerator(
        IDs_val,
        batch_size=args.bs_lower,
        dim=args.max_shift_len,
        vocab_size=embedding_matrix.shape[0],
        shuffle=True
    )
    test_generator = lowerDataGenerator(
        IDs_test,
        batch_size=args.bs_lower,
        dim=args.max_shift_len,
        vocab_size=embedding_matrix.shape[0],
        shuffle=False
    )

    return train_generator, valid_generator, test_generator


def get_generators_higher(args, embedding_matrix, IDs, df_survey, window=0):
    IDs_train, IDs_val, IDs_test = IDs

    train_generator = higherDataGenerator(
        df_survey, IDs_train,
        batch_size=args.bs_higher,
        dim=(args.max_n_shift, args.max_shift_len),
        vocab_size=embedding_matrix.shape[0],
        shuffle=True,
        if_deepsup=args.if_deepsup,
        if_taildrop=args.if_taildrop,
        max_tail=5,
        window=window
    )
    valid_generator = higherDataGenerator(
        df_survey, IDs_val,
        batch_size=1,
        dim=(args.max_n_shift, args.max_shift_len),
        vocab_size=embedding_matrix.shape[0],
        shuffle=False,
        if_deepsup=args.if_deepsup
    )
    test_generator = higherDataGenerator(
        df_survey, IDs_test,
        batch_size=1,
        dim=(args.max_n_shift, args.max_shift_len),
        vocab_size=embedding_matrix.shape[0],
        shuffle=False,
        if_deepsup=False
    )

    return train_generator, valid_generator, test_generator


def getGeneratorsSemi(args, embedding_matrix, IDs, df_survey, window=0):
    IDs_train, IDs_val, IDs_test = IDs

    train_generator = semiDataGenerator(
        df_survey, IDs_train, IDs_val,
        batch_size=1,
        dim=(args.max_n_shift, args.max_shift_len),
        vocab_size=embedding_matrix.shape[0],
        shuffle=True,
        if_deepsup=args.if_deepsup,
        if_taildrop=args.if_taildrop,
        max_tail=5,
        window=window,
        if_train=True
    )
    valid_generator = semiDataGenerator(
        df_survey, IDs_val, IDs_val,
        batch_size=1,
        dim=(args.max_n_shift, args.max_shift_len),
        vocab_size=embedding_matrix.shape[0],
        shuffle=False,
        if_deepsup=args.if_deepsup,
        if_train=False
    )
    test_generator = semiDataGenerator(
        df_survey, IDs_test, IDs_test,
        batch_size=1,
        dim=(args.max_n_shift, args.max_shift_len),
        vocab_size=embedding_matrix.shape[0],
        shuffle=False,
        if_deepsup=False,
        if_train=False
    )

    return train_generator, valid_generator, test_generator


def get_generator_single(args, embedding_matrix, IDs, df_survey):
    IDs_train, IDs_val, IDs_test = IDs
    train_generator = dataGeneratorSinLevel(
        df_survey, IDs_train,
        batch_size=args.batch_size,
        dim=args.max_n_actions,
        vocab_size=embedding_matrix.shape[0],
        shuffle=True,
        if_deepsup=args.if_deepsup
    )
    valid_generator = dataGeneratorSinLevel(
        df_survey, IDs_val,
        batch_size=1,
        dim=args.max_n_actions,
        vocab_size=embedding_matrix.shape[0],
        shuffle=False,
        if_deepsup=args.if_deepsup
    )
    test_generator = dataGeneratorSinLevel(
        df_survey, IDs_test,
        batch_size=1,
        dim=args.max_n_actions,
        vocab_size=embedding_matrix.shape[0],
        shuffle=False,
        if_deepsup=False
    )

    return train_generator, valid_generator, test_generator