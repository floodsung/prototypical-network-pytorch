import argparse

import torch
import torch.nn.functional as F
import torch.nn as nn

from torch.utils.data import DataLoader

from mini_imagenet import MiniImageNet
from samplers import CategoriesSampler
from senet import EmbeddingSENet,SEBasicBlock
from utils import pprint, set_gpu, count_acc, Averager, euclidean_metric
import numpy as np
import scipy as sp
import scipy.stats

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, h

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--load', default='./save/proto-1/max-acc.pth')
    parser.add_argument('--batch', type=int, default=600)
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)
    args = parser.parse_args()
    pprint(vars(args))

    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    dataset = MiniImageNet('test',is_train=False)
    sampler = CategoriesSampler(dataset.label,
                                args.batch, args.way, args.shot + args.query)
    loader = DataLoader(dataset, batch_sampler=sampler,
                        num_workers=8, pin_memory=True)

    model = EmbeddingSENet(SEBasicBlock,[3, 4, 6, 3],with_variation=True).to(device)
    model = nn.DataParallel(model,device_ids=[0,1,2,3])

    model.load_state_dict(torch.load(args.load))
    model.eval()

    ave_acc = Averager()
    accuracies = []

    for i, batch in enumerate(loader, 1):
        data, _ = [_.to(device) for _ in batch]
        k = args.way * args.shot
        data_shot, data_query = data[:k], data[k:]

        proto1,proto2,proto3,proto4,std_mean = model(data_shot)
        proto1 = proto1.reshape(args.shot, args.train_way, -1).mean(dim=0)
        proto2 = proto2.reshape(args.shot, args.train_way, -1).mean(dim=0)
        proto3 = proto3.reshape(args.shot, args.train_way, -1).mean(dim=0)
        proto4 = proto4.reshape(args.shot, args.train_way, -1).mean(dim=0)

        query1,query2,query3,query4,_ = model(data_query)

        logits_1 = euclidean_metric(query1, proto1)
        logits_2 = euclidean_metric(query2, proto2)
        logits_3 = euclidean_metric(query3, proto3)
        logits_4 = euclidean_metric(query4, proto4)
        logits = 0.3*logits_1+0.4*logits_2+0.5*logits_3+logits_4

        label = torch.arange(args.way).repeat(args.query)
        label = label.to(device)

        acc = count_acc(logits, label)
        ave_acc.add(acc)
        accuracies.append(acc)
        print('batch {}: {:.2f}({:.2f})'.format(i, ave_acc.item() * 100, acc * 100))

    m,h = mean_confidence_interval(accuracies)
    print("mean:",m,"h:",h)
