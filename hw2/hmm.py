# the python file is put into a sbfolder of the data folder;
# results are stored in result/ folder

import numpy as np
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

# inference of A
A = np.zeros([M, M])
for n in range(N):
    for t in range(sentence_len[n]-1):
        state_from = tag_dict[data_train[n][t][1]]
        state_to   = tag_dict[data_train[n][t+1][1]]
        A[state_from, state_to] += 1
from sklearn.preprocessing import normalize
A = normalize(A, axis=1, norm='l1')

# inference of pi
pi = np.zeros([M])
for n in range(N):
    state_init = tag_dict[data_train[n][0][1]]
    pi[state_init] += 1
pi = pi / np.sum(pi)

# inference of B
B = np.ones([M, V]) * 0.01
for n in range(N):
    for t in range(sentence_len[n]):
        word = vocab_dict[data_train[n][t][0]]
        state = tag_dict[data_train[n][t][1]]
        B[state, word] += 1
B = normalize(B, axis=1, norm='l1')

def neg_log_likelihood(data, pi, A, B, vocab_dict, tag_dict):
    M, V = B.shape
    N = data.shape[0]
    log_p_lst = np.zeros(N)
    sentence_len = list(map(len, data))
    for n in range(N):
        T = sentence_len[n]
        alpha = np.zeros([M, T])
        alpha_sum = np.zeros(T)
        word = vocab_dict[data[n][0][0]]
        alpha[:, 0] = pi * B[:, word]
        alpha_sum[0] = np.sum(alpha[:, 0])
        alpha[:, 0] = alpha[:, 0] / alpha_sum[0]
        for t in range(1, T):
            word = vocab_dict[data[n][t][0]]
            for j in range(M): 
                alpha[j, t] = np.sum([alpha[i, t-1] * A[i, j] * B[j, word] for i in range(M)])
            alpha_sum[t] = np.sum(alpha[:, t])
            alpha[:, t] = alpha[:, t] / alpha_sum[t]
        log_p_lst[n] = np.sum(-np.log(alpha_sum))
    return np.sum(log_p_lst)

p_train = neg_log_likelihood(data_train, pi, A, B, vocab_dict, tag_dict)
p_test = neg_log_likelihood(data_test, pi, A, B, vocab_dict, tag_dict)
transition = np.around(A,3)
# print(p_train, p_test)

# write results to another file
result = open('../result/hmm_result.txt', 'w')
result.write('Negative log likelihood on training set:' + str(np.around(p_train,1)) + '\n')
result.write('Negative log likelihood on test set:' + str(np.around(p_test,1)) + '\n')
result.write('Transition matrix A:\n') 
result.write(r'\begin{matrix}' + '\n')
result.write(' &' + ' &'.join(tagset)+ r'\\' + '\n')
for i in range(M):
    Ai_str = ' &'.join(map(str, transition[i,:]))
    result.write(tagset[i] + ' &' + Ai_str + r'\\' + '\n')
result.write(r'\end{matrix}')

state_pred = [None] * len(data_test)
for n in range(len(data_test)):
    sentence = data_test[n]
    T = len(sentence)
    V = np.zeros([M, T])
    BT = np.zeros([M, T])
    word = vocab_dict[sentence[0][0]]
    v_init = np.zeros([M])
    for i in range(M):
        v_init[i] = pi[i] * B[i, word] 
    V[:, 0] = v_init / np.sum(v_init)
    for t in range(1, T):
        word = vocab_dict[sentence[t][0]]
        for j in range(M):
            candidate = np.zeros([M])
            for i in range(M):
                candidate[i] = A[i, j] * B[j, word] * V[i, t-1]
            V[j, t] = np.max(candidate)
            BT[j, t] = np.argmax(candidate)
        V[:,t] = V[:, t] / np.sum(V[:, t])
    state_seq = np.zeros([T])
    state_seq[T-1] = np.argmax(V[:, T-1])
    for t in range(T-2, -1, -1):
        state_seq[t] = BT[int(state_seq[t+1]), t+1]
    state_pred[n] = state_seq

# accuracy
def acc(data_test, state_pred):
    N = len(data_test)
    accurate = 0
    total = 0
    for n in range(N):
        T = len(state_pred[n])
        for t in range(T):
            accurate += int(int(state_pred[n][t]) == tag_dict[data_test[n][t][1]])
            total += 1
    return accurate, total, np.around(accurate/total * 100, 2)

print('test set accuracy:')
print(np.around(acc(data_test, state_pred)[2], 2))

def confusion(data_test, state_pred):
    confusion = np.zeros([M, M])
    N = len(data_test)
    for n in range(N):
        T = len(state_pred[n])
        for t in range(T):
            confusion[int(state_pred[n][t]), tag_dict[data_test[n][t][1]]] += 1
    return confusion

cm = confusion(data_test, state_pred)
cm = cm.astype(int)

print(__doc__)

import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

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
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.tab20c)
plt.savefig('../result/cm_nonm.eps', format='eps', dpi=100)

plt.figure()
plot_confusion_matrix(cm, tagset,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.tab20c)
plt.savefig('../result/cm_nm.eps', format='eps', dpi=100)

