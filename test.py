import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mini_imagenet import MiniImageNet
from samplers import CategoriesSampler
from convnet import Convnet
from utils import pprint, set_gpu, count_acc, Averager, euclidean_metric


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--load', default='./save/proto-1/max-acc.pth')
    parser.add_argument('--batch', type=int, default=2000)
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=30)
    args = parser.parse_args()
    pprint(vars(args))

    set_gpu(args.gpu)

    dataset = MiniImageNet('test')
    sampler = CategoriesSampler(dataset.label,
                                args.batch, args.way, args.shot + args.query)
    loader = DataLoader(dataset, batch_sampler=sampler,
                        num_workers=8, pin_memory=True)

    model = Convnet().cuda()
    model.load_state_dict(torch.load(args.load))
    model.eval()

    ave_acc = Averager()

    for i, batch in enumerate(loader, 1):
        data, _ = [_.cuda() for _ in batch]
        k = args.way * args.shot
        data_shot, data_query = data[:k], data[k:]

        proto1,proto2,proto3,proto4 = model(data_shot)
        proto1 = proto1.reshape(args.shot, args.way, -1).mean(dim=0)
        proto2 = proto2.reshape(args.shot, args.way, -1).mean(dim=0)
        proto3 = proto3.reshape(args.shot, args.way, -1).mean(dim=0)
        proto4 = proto4.reshape(args.shot, args.way, -1).mean(dim=0)


        query1,query2,query3,query4 = model(data_query)

        logits1 = euclidean_metric(query1, proto1)
        logits2 = euclidean_metric(query2, proto2)
        logits3 = euclidean_metric(query3, proto3)
        logits4 = euclidean_metric(query4, proto4)
        logits = 0.3*logits1+0.4*logits2+0.5*logits3 + logits4
        

        label = torch.arange(args.way).repeat(args.query)
        label = label.type(torch.cuda.LongTensor)

        acc = count_acc(logits, label)
        ave_acc.add(acc)
        print('batch {}: {:.2f}({:.2f})'.format(i, ave_acc.item() * 100, acc * 100))

