# coding: utf-8

# In[1]:


import numpy as np
import torch
import torch.nn as nn

import sys

sys.path.append('/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages')

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


def get_hmm_params(data_train):
    V = len(vocab)
    M = len(tagset)
    N = data_train.shape[0]
    sentence_len = list(map(len, data_train))

    # inference of A
    A = np.zeros([M, M])
    for n in range(N):
        for t in range(sentence_len[n] - 1):
            state_from = tag_dict[data_train[n][t][1]]
            state_to = tag_dict[data_train[n][t + 1][1]]
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

    A_full = np.concatenate((A, pi.reshape((1, 12))))
    A_log = np.log(A_full)
    B_log = np.log(B)

    return (A_log, B_log)


# In[442]:


def to_tensor(numpy_array):
    # Numpy array -> Tensor
    return torch.from_numpy(numpy_array).float()


def to_variable(tensor):
    # Tensor -> Variable (on GPU if possible)
    if torch.cuda.is_available():
        # Tensor -> GPU Tensor
        tensor = tensor.cuda()
    return torch.autograd.Variable(tensor)


# In[452]:


def loader(data, batch_size=50):
    data_loader = []
    N = len(data)
    for batch_num in range(N // batch_size):
        batch = []
        for i in range(batch_size * batch_num, batch_size * (batch_num + 1)):
            document = data[i]
            x = to_tensor(np.array([vocab_dict[token[0]] for token in document]))
            y = to_tensor(np.array([tag_dict[token[1]] for token in document]))
            upper = to_tensor(np.array([int(sum([letter.isupper() for letter in word[0]]) == 1)
                                        for word in document]))
            x = to_variable(x.type('torch.LongTensor'))
            y = to_variable(y.type('torch.LongTensor'))
            upper = to_variable(upper.type('torch.LongTensor'))
            batch.append((x, y, upper))
        data_loader.append(batch)
    return data_loader


# train_loader = loader(data_train)[:5]


# In[524]:


def test_loader(data):
    data_loader = []
    N = len(data)
    for i in range(N):
        document = data[i]
        x = to_tensor(np.array([vocab_dict[token[0]] for token in document]))
        y = to_tensor(np.array([tag_dict[token[1]] for token in document]))
        upper = to_tensor(np.array([int(sum([letter.isupper() for letter in word[0]]) == 1)
                                    for word in document]))
        x = to_variable(x.type('torch.LongTensor'))
        y = to_variable(y.type('torch.LongTensor'))
        upper = to_variable(upper.type('torch.LongTensor'))
        data_loader.append((x, y, upper))
    return data_loader


# test_loader = test_loader(data_test)



# In[518]:


# original HMM model applied to CRF
class CRF(nn.Module):
    def __init__(self, M, V, T=None, E=None):
        super().__init__()
        self.M = M
        self.V = V
        if T is None:
            self.T = nn.Parameter(torch.zeros(M + 1, M).type('torch.DoubleTensor'))
        else:
            self.T = nn.Parameter(to_tensor(T).type('torch.DoubleTensor'))
        if E is None:
            self.E = nn.Parameter(torch.zeros(M, V).type('torch.DoubleTensor'))
        else:
            self.E = nn.Parameter(to_tensor(E).type('torch.DoubleTensor'))
        self.Eprev = nn.Parameter(torch.zeros(M, V + 1).type('torch.DoubleTensor'))
        self.Enext = nn.Parameter(torch.zeros(M, V + 1).type('torch.DoubleTensor'))
        self.Cap = nn.Parameter(torch.zeros(M, 2).type('torch.DoubleTensor'))

    def log_z(self, x, upper):
        log_z = 0
        M = self.M
        init = to_variable(torch.LongTensor([M]))
        alpha = to_variable(torch.zeros([M]).type('torch.DoubleTensor'))
        for i in range(self.M):
            alpha[i] = torch.exp(self.log_phi(to_variable(torch.LongTensor([i])),
                                              init, x, 0, upper))
        beta = alpha.clone() / sum(alpha.clone())
        log_z += torch.log(sum(alpha.clone()))
        for t in range(1, len(x)):
            for i in range(self.M):
                alpha[i] = torch.dot(beta.clone(), self.T[0:self.M, i]) * \
                           self.E[to_variable(torch.LongTensor([i])), x[t]]
            beta = alpha.clone() / sum(alpha.clone())
            log_z += torch.log(sum(alpha.clone()))
        return log_z

    def log_phi(self, y, y_prev, x, t, upper):
        if t == 0:
            x_prev = to_variable(torch.LongTensor([M]))
        else:
            x_prev = x[t - 1]
        if t == len(x) - 1:
            x_next = to_variable(torch.LongTensor([M]))
        else:
            x_next = x[t + 1]
        return (torch.index_select(torch.index_select(self.T, 0, y_prev), 1, y) +
                torch.index_select(torch.index_select(self.Eprev, 0, y), 1, x_prev) +
                torch.index_select(torch.index_select(self.Enext, 0, y), 1, x_next) +
                torch.index_select(torch.index_select(self.Cap, 0, y), 1, upper[t]) +
                torch.index_select(torch.index_select(self.E, 0, y), 1, x[t])).squeeze()

    def forward(self, x, y, upper):
        p_log = self.log_z(x, upper)
        # p_log = 0
        init = to_variable(torch.LongTensor([self.M]))
        p_log += - self.log_phi(y[0], init, x, 0, upper)
        for i in range(1, len(y)):
            p_log += - self.log_phi(y[i], y[i - 1], x, i, upper)
        # print(p_log.data)
        return p_log

    def prediction(self, x, upper):
        n = len(x)
        pred_y = [0] * n
        v = np.zeros([self.M, len(x)])
        bt = np.zeros([self.M, len(x)])
        init = to_variable(torch.LongTensor([self.M]))
        for i in range(self.M):
            # print(self.log_phi(to_variable(torch.LongTensor([i])), init, x, 0).data.float())
            v[i, 0] = self.log_phi(to_variable(torch.LongTensor([i])), init,
                                   x, 0, upper).data.numpy()
        for t in range(1, n):
            for i in range(self.M):
                log_p_seq = [v[j, t - 1] + self.log_phi(to_variable(torch.LongTensor([i])),
                                                        to_variable(torch.LongTensor([j])),
                                                        x, t, upper).data.numpy()
                             for j in range(self.M)]
                v[i, t] = max(log_p_seq)
                bt[i, t] = np.argmax(log_p_seq)
        pred_y[n - 1] = np.argmax(v[:, n - 1])
        for t in range(n - 2, -1, -1):
            pred_y[t] = int(bt[int(pred_y[t + 1]), t + 1])
        return pred_y


# In[520]:


def crf_train(my_net, num_epoch, train_loader):
    print('Beginning training:')
    optimizer = torch.optim.Adam(my_net.parameters(), lr=0.0001)
    for epoch in range(num_epoch):
        i = 0
        for batch in train_loader:
            loss = 0
            for (x, y, upper) in batch:
                loss += my_net(x, y, upper)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            i += 1
            print('epoch %d batch %d train loss: %.3f.' % (epoch, i, loss))
    return my_net


# trained_net = crf_train(model, 1, train_loader, A_log = A_log, B_log = B_log)
# torch.save(trained_net.state_dict(), 'param.pt')


# In[528]:


def acc(trained_net, test_loader):
    print('Calculating accuracy rate:')
    accurate = 0
    total = 0
    i = 0
    for x, y, upper in test_loader:
        T = len(x)
        pred = trained_net.prediction(x, upper)
        for t in range(T):
            accurate += int(pred[t] == y.data[t])
            total += 1
        i += 1
        print(i, end=',')
    print(accurate, total, accurate / total)
    return accurate, total, accurate / total


if __name__ == '__main__':
    import numpy as np

    data_train = np.load("../train_set.npy")
    data_test = np.load("../test_set.npy")
    vocab = open('../vocab.txt').read().splitlines()
    tagset = open('../tagset.txt').read().splitlines()
    V = len(vocab)
    M = len(tagset)
    N = data_train.shape[0]

    train_loader = loader(data_train)
    test_loader = test_loader(data_test)
    A_log, B_log = get_hmm_params(data_train)

    my_net = CRF(M, V, T=A_log, E=B_log)
    my_net.load_state_dict(torch.load('param.pt'))
    my_net.eval()
    trained_net = crf_train(my_net, 1, train_loader)
    print('Saving model parameters:')
    torch.save(trained_net.state_dict(), 'param2.pt')
    acc(trained_net, test_loader)
