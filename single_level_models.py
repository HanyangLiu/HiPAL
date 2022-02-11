import heapq

import pandas as pd
import numpy as np
import os
import pickle
from sklearn import metrics
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
import argparse
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model
import neptune
from neptunecontrib.monitoring import utils
from neptunecontrib.monitoring.keras import NeptuneMonitor
from utils.util_models import getEpochNum, PredictiveModel, InnerPretrain
from utils.util_single_level import singleLevelModel, dataGeneratorSinLevelVariantHorizon
from utils.util_setup import *
from utils.util_evaluation import evaluate
from informer.informer_predictor import InformerPredictor


def train_outer_model(train_generator, test_generator, args):
    '''Fit model'''
    model_family = singleLevelModel(
        max_n_actions=args.max_n_actions,
        embedding_matrix=embedding_matrix,
        dropout_rate=0.5
    )
    if args.model_name == 'simpletcn':
        model = model_family.simpletcn()
    elif args.model_name == 'restcn':
        model = model_family.restcn()
    elif args.model_name == 'gru':
        model = model_family.gru()
    elif args.model_name == 'fcn':
        model = model_family.fcn()
    else:
        ValueError("Please specify model name!")

    print(args)
    # define the checkpoints
    monitor = 'val_auc'
    checkpoint = ModelCheckpoint(args.outer_save_path, monitor=monitor, verbose=1, save_best_only=True, mode='max')
    earlystopping = EarlyStopping(monitor=monitor, mode='max', verbose=1, patience=20)
    callback_list = [checkpoint]
    if args.if_earlystopping: callback_list.append(earlystopping)
    if args.if_neptune:  callback_list += [NeptuneMonitor(), getEpochNum()]
    # train model
    model.fit_generator(generator=train_generator,
                        validation_data=test_generator,
                        epochs=args.epochs,
                        use_multiprocessing=True,  #int(args.gpu_id) == -1,  # multiple CPUs for data preprocessing
                        workers=args.workers,  # multiple CPUs for calculating
                        callbacks=callback_list,
                        verbose=1)


def evaluateVariantHorizon(best_model, args, df_survey, IDs_test, embedding_matrix, horizon=0):
    predictor = keras.models.Model(inputs=best_model.input, outputs=best_model.get_layer('prediction').output)
    predictor.compile(
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(curve='ROC'), tf.keras.metrics.AUC(curve='PR')]
    )

    print('-----------Test horizon {} -----------'.format(horizon))
    test_generator = dataGeneratorSinLevelVariantHorizon(df_survey, IDs_test,
                                                         batch_size=1,
                                                         dim=args.max_n_actions,
                                                         vocab_size=embedding_matrix.shape[0],
                                                         shuffle=False,
                                                         if_deepsup=False,
                                                         horizon=horizon)
    # Get testing scores
    print('Testing model...')
    y_test = []
    for i in range(len(test_generator)):
        y_i = test_generator[i][1][:, 1]
        y_test += list(y_i)
    y_test = np.array(y_test)
    y_prob = predictor.predict_generator(test_generator, steps=len(test_generator), verbose=1)[:, 1]
    # print(y_prob)

    # evaluation
    fpr, tpr, _ = metrics.roc_curve(y_test, y_prob)
    prec, rec, _ = metrics.precision_recall_curve(y_test, y_prob)
    acc = metrics.accuracy_score(y_test, y_prob > 0.5)
    print('--------------------------------------------')
    print('Evaluation of full test set:')
    print("AU-ROC:", "%0.4f" % metrics.auc(fpr, tpr),
          "AU-PRC:", "%0.4f" % metrics.auc(rec, prec),
          "Accuracy:", "%0.4f" % acc)

    if args.if_neptune:
        neptune.log_metric('AU-ROC', metrics.auc(fpr, tpr))
        neptune.log_metric('AU-PRC', metrics.auc(rec, prec))
        neptune.log_metric('Accuracy', acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--memo', type=str, default='CV_single_resTCN_woActEmb')
    parser.add_argument('--max_n_actions', type=int, default=50000)
    parser.add_argument('--dim_embedding', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=5)
    parser.add_argument('--tcn_layers', type=int, default=6)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--if_pretrain', type=boolean_string, default=False)
    parser.add_argument('--if_deepsup', type=boolean_string, default=False)
    parser.add_argument('--if_train_predictor', type=boolean_string, default=False)
    parser.add_argument('--if_neptune', type=boolean_string, default=False)
    parser.add_argument('--if_earlystopping', type=boolean_string, default=True)
    parser.add_argument('--if_mem_constr', type=boolean_string, default=True)
    parser.add_argument('--if_flatten', type=boolean_string, default=True)
    parser.add_argument('--if_static', type=boolean_string, default=False)
    parser.add_argument('--trial_idx', type=int, default=5)
    parser.add_argument('--cv_idx', type=int, default=0)
    parser.add_argument('--model_name', type=str, default='tcn')
    parser.add_argument('--horizon', type=int, default=5)
    args = parser.parse_args()
    print(args)

    # Use GPU for training
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    if args.if_mem_constr:
        physical_devices = tf.config.list_physical_devices('GPU')
        try: tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except: pass

    # log experiment with Neptune and set random seed
    if args.if_neptune: init_neptune(args)
    set_random_seeds(123)

    # load survey and embeddings, make dir
    survey = pd.read_csv('data_processed/survey_processed.csv')
    survey = survey[survey['burnout_bin'].notna()]
    embedding_matrix = get_embeddings(args.dim_embedding)

    # model path
    model_folder = os.path.join('model', args.memo)
    path_suffix = '_trial_' + str(args.trial_idx) + '_fold_' + str(args.cv_idx)
    if not os.path.exists(model_folder): os.mkdir(model_folder)
    args.outer_save_path = os.path.join(model_folder, "higher" + path_suffix)

    # train-validation split
    print('\n--------- Trial index: {} ---------'.format(args.trial_idx))
    print('\nFold index (0-4): {}'.format(args.cv_idx))
    cv = RandomizedGroupKFold(groups=survey['USER_ID'].to_numpy(), n_splits=5, random_state=args.trial_idx)
    train_ix, test_ix = cv[args.cv_idx]
    train_uids = survey.iloc[train_ix, :]['USER_ID'].unique().tolist()
    test_uids = survey.iloc[test_ix, :]['USER_ID'].unique().tolist()


    # prepare data for higher model
    _, df_survey, IDs = data_loader_outer(train_uids, test_uids)
    print('------------------\nIncluded surveys:', df_survey.sample_ID.nunique(), '\n-------------------')
    train_generator, valid_generator, test_generator = get_generator_single(args, embedding_matrix, IDs, df_survey)
    # train higher model
    if args.if_train_predictor: train_outer_model(train_generator, valid_generator, args)
    best_model = load_model(args.outer_save_path)
    evaluate(best_model, test_generator, args)
    # evaluateVariantHorizon(best_model, args, df_survey, IDs[-1], embedding_matrix, horizon=args.horizon)




