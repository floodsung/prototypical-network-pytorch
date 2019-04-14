import os
import shutil
import time
import pprint

import torch


def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)


def ensure_path(path):
    if os.path.exists(path):
        if input('{} exists, remove? ([y]/n)'.format(path)) != 'n':
            shutil.rmtree(path)
            os.mkdir(path)
    else:
        os.mkdir(path)


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    return (pred == label).type(torch.cuda.FloatTensor).mean().item()


def dot_metric(a, b):
    return torch.mm(a, b.t())


def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits

def wasserstein_metric(a,b):
    _,M,D = a.shape
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(-1,m,-1,-1).view(-1,M,D)
    b = b.unsqueeze(0).expand(n,-1,-1,-1).view(-1,M,D)
    logits = -wasserstein_distance(a,b).view(n,m)
    return logits


def wasserstein_distance(X,Y):
    # shape of a and b: [B,M,512]
    B,M,D = a.shape

    cost = torch.pairwise_distance(X.unsqueeze(2).expand(-1,-1,M,-1).reshape((-1, D)), 
                                       Y.unsqueeze(1).expand(-1,M,-1,-1).reshape((-1, D))
                                      ).reshape((B, M, M))
    m = -cost/0.1
        
    for i in range(10):
        m = m - m.logsumexp(dim=2, keepdim=True)
        m = m - m.logsumexp(dim=1, keepdim=True)
            
    m = m.softmax(dim=2)

    dist = torch.diagonal(torch.matmul(m.permute(0,2,1), cost), dim1=-2, dim2=-1).sum(dim=1)

    return dist


class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)


def l2_loss(pred, label):
    return ((pred - label)**2).sum() / len(pred) / 2

