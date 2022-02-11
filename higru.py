import argparse
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model
import neptune
from neptunecontrib.monitoring import utils
from neptunecontrib.monitoring.keras import NeptuneMonitor
from utils.util_models import getEpochNum
from utils.util_higru import buildHierGRU, inner_ce_loss
from utils.util_setup import *
from utils.util_evaluation import evaluate, evaluateVariantHorizon
from utils.util_data_v0 import boolean_string


def train_inner_model(train_generator, test_generator, ARGS):
    '''Fit model'''
    model = ARGS.built_model.build_lower_model()
    print(ARGS)
    model.summary()
    # define the checkpoints
    checkpoint = ModelCheckpoint(args.inner_save_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    earlystopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
    callback_list = [checkpoint]
    if ARGS.if_earlystopping: callback_list.append(earlystopping)
    if ARGS.if_neptune:  callback_list += [NeptuneMonitor(), getEpochNum()]
    # train model
    model.fit_generator(generator=train_generator,
                        validation_data=test_generator,
                        epochs=ARGS.epochs_lower,
                        use_multiprocessing=True,  # multiple CPUs for data preprocessing
                        workers=ARGS.workers,  # multiple CPUs for calculating
                        callbacks=callback_list,
                        verbose=1)


def train_outer_model(train_generator, test_generator, ARGS):
    '''Fit model'''
    model = ARGS.built_model.build_higher_model()

    # load pretrained weights
    if ARGS.if_pretrain:
        lower_model = load_model(ARGS.inner_save_path)
        pretrained_weights = lower_model.get_layer('nested_lower_encoding').get_weights()
        model.get_layer('nested_base_model').layer.set_weights(pretrained_weights)
        print("Pretrained lower model weights loaded.")

    print(ARGS)
    model.summary()
    # define the checkpoints
    monitor = 'val_prediction_auc' if args.if_deepsup else 'val_auc'
    checkpoint = ModelCheckpoint(args.outer_save_path, monitor=monitor, verbose=1, save_best_only=True, mode='max')
    earlystopping = EarlyStopping(monitor=monitor, mode='max', verbose=1, patience=20)
    callback_list = [checkpoint]
    if ARGS.if_earlystopping: callback_list.append(earlystopping)
    if ARGS.if_neptune:  callback_list += [NeptuneMonitor(), getEpochNum()]
    # train model
    model.fit_generator(generator=train_generator,
                        validation_data=test_generator,
                        epochs=ARGS.epochs_higher,
                        use_multiprocessing=True, #int(args.gpu_id) == -1,  # multiple CPUs for data preprocessing
                        workers=ARGS.workers,  # multiple CPUs for calculating
                        callbacks=callback_list,
                        verbose=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--memo', type=str, default='testing_gru')
    parser.add_argument('--max_n_shift', type=int, default=30)
    parser.add_argument('--max_shift_len', type=int, default=3008)
    parser.add_argument('--dim_embedding', type=int, default=100)
    parser.add_argument('--epochs_lower', type=int, default=10)
    parser.add_argument('--epochs_higher', type=int, default=10)
    parser.add_argument('--bs_lower', type=int, default=16)
    parser.add_argument('--bs_higher', type=int, default=4)
    parser.add_argument('--workers', type=int, default=5)
    parser.add_argument('--tcn_layers', type=int, default=6)
    parser.add_argument('--time_size', type=int, default=50)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--if_pretrain', type=boolean_string, default=False)
    parser.add_argument('--pretrain_model', type=str, default='testing_gru')
    parser.add_argument('--if_train_predictor', type=boolean_string, default=True)
    parser.add_argument('--if_neptune', type=boolean_string, default=False)
    parser.add_argument('--if_earlystopping', type=boolean_string, default=True)
    parser.add_argument('--if_mem_constr', type=boolean_string, default=True)
    parser.add_argument('--if_deepsup', type=boolean_string, default=True)
    parser.add_argument('--if_taildrop', type=boolean_string, default=True)
    parser.add_argument('--if_static', type=boolean_string, default=False)
    parser.add_argument('--trial_idx', type=int, default=5)
    parser.add_argument('--cv_idx', type=int, default=0)
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
    args.inner_save_path = 'model/' + args.pretrain_model + '/lower_all'
    args.outer_save_path = os.path.join(model_folder, "higher" + path_suffix)

    # train-validation split
    print('\n--------- Trial index: {} ---------'.format(args.trial_idx))
    print('\nFold index (0-4): {}'.format(args.cv_idx))
    cv = RandomizedGroupKFold(groups=survey['USER_ID'].to_numpy(), n_splits=5, random_state=args.trial_idx)
    train_ix, test_ix = cv[args.cv_idx]
    train_uids = survey.iloc[train_ix, :]['USER_ID'].unique().tolist()
    val_uids = survey.iloc[test_ix, :]['USER_ID'].unique().tolist()

    # define model
    args.built_model = buildHierGRU(args, embedding_matrix)

    if args.if_pretrain and not os.path.exists(args.inner_save_path):
        # prepare data for lower model
        os.mkdir(os.path.join('model', args.pretrain_model))
        _, _, IDs = data_loader_inner(train_uids, val_uids)
        train_generator, valid_generator, _ = get_generators_lower(args, embedding_matrix, IDs)

        # train lower model
        print("Training lower model ...")
        train_inner_model(train_generator, valid_generator, args)

    # prepare data for higher model
    _, df_survey, IDs = data_loader_outer(train_uids, val_uids)
    print('------------------\nIncluded surveys:', df_survey.sample_ID.nunique(), '\n-------------------')
    train_generator, valid_generator, test_generator = get_generators_higher(args, embedding_matrix, IDs, df_survey)

    # train higher model
    if args.if_train_predictor:
        print("Training higher model ...")
        train_outer_model(train_generator, valid_generator, args)

    best_model = load_model(args.outer_save_path, custom_objects={"inner_ce_loss": inner_ce_loss})
    evaluate(best_model, test_generator, args)




