import pandas as pd
import numpy as np
from gensim.models import Word2Vec, KeyedVectors
from gensim.models.callbacks import CallbackAny2Vec
import pickle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.cm as cm
import matplotlib.pyplot as plt


# define callback
class LossLogger(CallbackAny2Vec):
    '''Output loss at each epoch'''
    def __init__(self):
        self.epoch = 1
        self.losses = []

    def on_epoch_begin(self, model):
        print(f'Epoch: {self.epoch}', end='\t')

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        self.losses.append(loss)
        print(f'  Loss: {loss}')
        self.epoch += 1


class callback(CallbackAny2Vec):
    """
    Callback to print loss after each epoch
    """
    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        if self.epoch == 0:
            print('Loss after epoch {}: {}'.format(self.epoch, loss))
        else:
            print('Loss after epoch {}: {}'.format(self.epoch, loss - self.loss_previous_step))
        self.epoch += 1
        self.loss_previous_step = loss


def train_embeddings():
    # # train embeddings
    # loss_logger = LossLogger()
    # model = Word2Vec(sentences=corpus,
    #                  size=embedding_size,
    #                  window=context,
    #                  min_count=min_count,
    #                  workers=10,
    #                  sg=1,
    #                  callbacks=[callback()],
    #                  iter=epochs)

    # init word2vec class
    model = Word2Vec(min_count=min_count,
                     window=context,
                     size=embedding_size,
                     workers=10)
    # build vovab
    model.build_vocab(corpus)

    # train the w2v model
    model.train(sentences=corpus,
                total_examples=model.corpus_count,
                epochs=epochs,
                report_delay=1,
                compute_loss=True,  # set compute_loss = True
                callbacks=[callback()])  # add the callback class

    # save model
    wv = model.wv
    wv.save(word_vector_file)


def visualize():
    wv = KeyedVectors.load(word_vector_file)
    logs_all = pd.read_csv(logs_file)

    top_counts = logs_all['metric_ID'].value_counts().head(50)
    print(top_counts)
    tokens = top_counts.index.to_list()
    word_set = set()
    for token in tokens:
        print(token, token_dict[str(token)], top_counts[token])
        top = wv.most_similar(positive=[str(token)], topn=5)
        top_list = [(x[0], token_dict[x[0]], x[1]) for x in top]
        # print(top_list, '\n')
        word_set = word_set | set([x[0] for x in top_list])
    word_list = list(word_set)
    X = wv[word_list]
    pca = PCA(n_components=2)
    X_2 = pca.fit_transform(X)
    model_tsne = TSNE(n_components=2,
                      init='pca',
                      random_state=123,
                      method='barnes_hut',
                      perplexity=40,
                      n_iter=1000,
                      verbose=2)
    Y = model_tsne.fit_transform(X)
    labels = [token_dict[x] for x in word_list]

    # Show the scatter plot
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], 20)
    # Add labels
    for label, x, y in zip([x[:4] for x in labels], X[:, 0], X[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points', size=8, rotation=35)
    plt.title('PCA of action embedding')
    plt.savefig('result/pca_2.pdf')
    plt.show()

    # Show the scatter plot
    plt.scatter(X[:, 0], X[:, 1], 20)
    # Add labels
    for label, x, y in zip(word_list, X[:, 0], X[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points', size=8, rotation=35)
    plt.title('PCA of action embedding')
    plt.savefig('result/pca_1.pdf')
    plt.show()

    tokens = list(map(str, range(1, 51)))
    word_set = set()
    for token in tokens:
        print('Token:', token, token_dict[str(token)])
        top = wv.most_similar(positive=[str(token)], topn=10)
        top_list = [(x[0], token_dict[x[0]], x[1]) for x in top]
        print(top_list, '\n')
        word_set = word_set | set([x[0] for x in top_list])
    word_list = list(word_set)
    X = wv[word_list]
    pca = PCA(n_components=2)
    X_2 = pca.fit_transform(X)
    model_tsne = TSNE(n_components=2, random_state=0)
    Y = model_tsne.fit_transform(X)
    labels = [token_dict[x] for x in word_list]

    # Show the scatter plot
    plt.scatter(Y[:, 0], Y[:, 1], 20)
    # Add labels
    for label, x, y in zip([x[:3] for x in labels], Y[:, 0], Y[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points', size=8, rotation=35)
    plt.title('TSNE of action embedding')
    plt.savefig('result/tsne.pdf')
    plt.show()


    label_list = [label[:5] for label in labels]
    colors = cm.rainbow(np.linspace(0, 1, len(label_list)))
    # Show the scatter plot
    plt.figure(figsize=(7, 7))
    for l, c, co in zip(label_list, colors, range(len(label_list))):
        plt.scatter(Y[np.where(np.array(label_list) == l), 0],
                    Y[np.where(np.array(label_list) == l), 1],
                    marker='o',
                    color=c,
                    linewidth=1.5,
                    alpha=0.8,
                    label=l)
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off
    plt.tick_params(
        axis='y',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        left=False,  # ticks along the bottom edge are off
        right=False,  # ticks along the top edge are off
        labelleft=False)  # labels along the bottom edge are off
    # plt.savefig('result/tsne.pdf')
    # plt.legend(loc='best')
    plt.show()





    num_embedding = 1000
    action_list = df_metric_cat.iloc[0:num_embedding]['action'].values.tolist()
    inv_token_dict = {v: k for k, v in token_dict.items()}
    word_list = [inv_token_dict[action_list[x]] for x in range(num_embedding) if action_list[x] in inv_token_dict]
    type_list = np.array([metric_dict[action_list[x]] for x in range(num_embedding) if action_list[x] in inv_token_dict])
    metric_list = df_metric_cat.iloc[0:num_embedding]['METRIC_NAME'].apply(lambda x: x[0:5])

    X = wv[word_list]
    pca = PCA(n_components=2)
    X_2 = pca.fit_transform(X)
    model_tsne = TSNE(n_components=2,
                      init='pca',
                      random_state=123,
                      method='barnes_hut',
                      perplexity=40,
                      n_iter=1000,
                      verbose=2)
    Y = model_tsne.fit_transform(X)

    label_list = df_metric_cat.iloc[0:num_embedding]['metric_category'].unique().tolist()
    label_list = label_list[:8] + label_list[9:]
    colors = cm.rainbow(np.linspace(0, 1, len(label_list)))

    # Show the scatter plot
    plt.figure(figsize=(7, 7))
    l = 'Chart Review'
    plt.scatter(Y[np.where(type_list == l), 0],
                Y[np.where(type_list == l), 1],
                marker='o',
                color='lightsteelblue',
                linewidth=1,
                alpha=0.8,
                label=l)
    for l, c, co in zip(label_list, colors, range(len(label_list))):
        if l == 'Chart Review':
            continue
        plt.scatter(Y[np.where(type_list == l), 0],
                    Y[np.where(type_list == l), 1],
                    marker='o',
                    color=c,
                    linewidth=1.5,
                    alpha=0.8,
                    label=l)
    l = 'Unknown'
    plt.scatter(Y[np.where(type_list == l), 0],
                Y[np.where(type_list == l), 1],
                marker='o',
                color='black',
                linewidth=1.5,
                alpha=0.8,
                label=l)
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off
    plt.tick_params(
        axis='y',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        left=False,  # ticks along the bottom edge are off
        right=False,  # ticks along the top edge are off
        labelleft=False)  # labels along the bottom edge are off
    plt.savefig('result/tsne.pdf')
    plt.legend(loc='best')
    plt.show()


if __name__ == "__main__":
    data_dir = 'IGNITE/data_output_from_SSIS_2'
    access_file_name = 'access_log_complete.csv'
    corpus_file = "data_processed/corpus_metric_name.pkl"
    dict_file = 'data_processed/token_dict_metric_name.pkl'
    metric_cat = 'data_processed/metric_categorized.csv'
    logs_file = 'data_processed/logs_all.csv'

    # parameters
    context = 10
    embedding_size = 100
    min_count = 1
    epochs = 20
    if_train = False
    if_visualize = True
    word_vector_file = 'data_processed/metric_vectors_' + str(embedding_size) + '.wv'

    with open(corpus_file, 'rb') as f:
        corpus = pickle.load(f)
    with open(dict_file, 'rb') as f:
        token_dict = pickle.load(f)

    df_metric_cat = pd.read_csv(metric_cat)
    # df_metric_cat['action'] = df_metric_cat['METRIC_NAME'].map(str) + '-' + df_metric_cat['REPORT_NAME'].map(str)
    df_metric_cat['action'] = df_metric_cat['METRIC_NAME'].map(str)
    metric_dict = dict(zip(df_metric_cat['action'].values, df_metric_cat['metric_category']))

    # train action embeddings
    if if_train:
        train_embeddings()
    if if_visualize:
        visualize()


