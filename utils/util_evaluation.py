from tensorflow import keras
from sklearn import metrics
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import neptune
from neptunecontrib.monitoring import utils
from utils.util_data_v0 import DataGeneratorVariantHorizon


def evaluate(best_model, test_generator, args):
    predictor = keras.models.Model(inputs=best_model.input, outputs=best_model.get_layer('prediction').output)
    predictor.compile(
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(curve='ROC'), tf.keras.metrics.AUC(curve='PR')]
    )
    loss, acc, auroc, auprc = predictor.evaluate(test_generator, verbose=1)
    print('--------------------------------------------')
    print('Evaluation of full test set (by model.evalulate()):')
    print("AU-ROC:", "%0.4f" % auroc,
          "AU-PRC:", "%0.4f" % auprc,
          "Accuracy:", "%0.4f" % acc)

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
    f1 = metrics.f1_score(y_test, y_prob > 0.5)
    print('--------------------------------------------')
    print('Evaluation of full test set:')
    print("AU-ROC:", "%0.4f" % metrics.auc(fpr, tpr),
          "AU-PRC:", "%0.4f" % metrics.auc(rec, prec),
          "Accuracy:", "%0.4f" % acc,
          "F1:", "%0.4f" % f1,)

    if args.if_neptune:
        neptune.log_metric('AU-ROC', metrics.auc(fpr, tpr))
        neptune.log_metric('AU-PRC', metrics.auc(rec, prec))
        neptune.log_metric('Accuracy', acc)
        neptune.log_metric('F1', f1)
        # plot ROC and PRC
        fig1 = plt.figure()
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, label='AUC = %0.4f' % metrics.auc(fpr, tpr))
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        utils.send_figure(fig1, channel_name='figures')

        fig2 = plt.figure()
        plt.title('Precision Recall Curve')
        plt.plot(rec, prec, label='AUC = %0.4f' % metrics.auc(rec, prec))
        plt.legend(loc='lower right')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.ylabel('Precision')
        plt.xlabel('Recall')
        utils.send_figure(fig2, channel_name='figures')

    return metrics.auc(fpr, tpr), metrics.auc(rec, prec), acc, f1


def evaluateVariantHorizon(best_model, args, df_survey, IDs_test, embedding_matrix, horizon=0):
    predictor = keras.models.Model(inputs=best_model.input, outputs=best_model.get_layer('prediction').output)
    predictor.compile(
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(curve='ROC'), tf.keras.metrics.AUC(curve='PR')]
    )

    print('-----------Test horizon {} -----------'.format(horizon))
    test_generator = DataGeneratorVariantHorizon(df_survey, IDs_test,
                                                 batch_size=1,
                                                 dim=(args.max_n_shift, args.max_shift_len),
                                                 vocab_size=embedding_matrix.shape[0],
                                                 shuffle=False,
                                                 if_deepsup=False,
                                                 horizon=horizon,
                                                 if_rand_horizon=False)
    # loss, acc, auroc, auprc = predictor.evaluate(test_generator, verbose=1)
    # print('--------------------------------------------')
    # print('Evaluation of full test set (by model.evalulate()):')
    # print("AU-ROC:", "%0.4f" % auroc,
    #       "AU-PRC:", "%0.4f" % auprc,
    #       "Accuracy:", "%0.4f" % acc)

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

