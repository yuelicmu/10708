from crf_main import CRF, get_hmm_params, loader, test_loader
import numpy as np
import sys

sys.path.append('/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages')
sys.path.append('/Users/yueli/src/tqdm')

import torch
import torch.nn as nn
from tqdm import tqdm

LOAD_PATH = 'param15.pt'

data_train = np.load("../train_set.npy")
data_test = np.load("../test_set.npy")

vocab = open('../vocab.txt').read().splitlines()
tagset = open('../tagset.txt').read().splitlines()

tag_dict = dict(zip(tagset, range(12)))
vocab_dict = dict(zip(vocab, range(len(vocab))))

V = len(vocab)
M = len(tagset)
N = data_train.shape[0]
sentence_len = list(map(len, data_train))

train_loader = test_loader(data_train)
test_loader = test_loader(data_test)
A_log, B_log = get_hmm_params(data_train)

my_net = CRF(M, V, T=A_log, E=B_log)
my_net.load_state_dict(torch.load(LOAD_PATH))
my_net.eval()

# negative log likelihood on test set
log_p = 0
for i in tqdm(range(len(test_loader))):
# for x, y, upper in test_loader:
    x, y, upper = test_loader[i]
    log_p += my_net(x, y, upper).data.numpy()
print('Negative log-likelihood on test set:')
print(log_p)

# negative log likelihood on train set
log_p = 0
for i in tqdm(range(len(train_loader))):
# for x, y, upper in test_loader:
    x, y, upper = train_loader[i]
    log_p += my_net(x, y, upper).data.numpy()
print('Negative log-likelihood on training set:')
print(log_p)

# accuracy function
def acc(trained_net, test_loader):
    print('Calculating accuracy rate:')
    accurate = 0
    total = 0
    i = 0
    N = len(test_loader)
    for i in tqdm(range(N)):
        x, y, upper = test_loader[i]
        T = len(x)
        pred = trained_net.prediction(x, upper)
        for t in range(T):
            accurate += int(pred[t] == y.data[t])
            total += 1
    print(accurate, total, accurate / total)
    return accurate, total, accurate / total
acc(my_net, test_loader)

# calculating confusion matrix
def confusion(trained_net, test_loader, M=12):
    print('Calculating confusion matrix:')
    confusion = np.zeros([M, M])
    N = len(test_loader)
    for i in tqdm(range(N)):
        x, y, upper = test_loader[i]
        T = len(x)
        pred = trained_net.prediction(x, upper)
        for t in range(T):
            confusion[int(pred[t]), int(y[t])] += 1
    np.save('comfusion.npy', confusion)
    print('Confusion matrix saved.')
    return confusion

cm = confusion(my_net, test_loader)
cm = cm.astype(int)


import matplotlib.pyplot as plt
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    #else:
    #    print('Confusion matrix, without normalization')

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 fontsize = 6,
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


plt.figure()
plot_confusion_matrix(cm, tagset,
                          normalize=True,
                          title='Confusion matrix for CRF')
plt.savefig('../cm.eps', format='eps', dpi=100)
