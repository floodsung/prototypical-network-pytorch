import argparse
import os.path as osp

import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.utils.data import DataLoader

from mini_imagenet import MiniImageNet
from samplers import CategoriesSampler
from senet import EmbeddingSENet,SEBasicBlock
from utils import pprint, set_gpu, ensure_path, Averager, Timer, count_acc, euclidean_metric


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=100)
    parser.add_argument('--save-epoch', type=int, default=10)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--train-way', type=int, default=30)
    parser.add_argument('--test-way', type=int, default=5)
    parser.add_argument('--save-path', default='./save/proto-1')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    pprint(vars(args))

    ensure_path(args.save_path)
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')


    trainset = MiniImageNet('train_val')
    train_sampler = CategoriesSampler(trainset.label, 1000,
                                      args.train_way, args.shot + args.query)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler,
                              num_workers=8, pin_memory=True)

    model = EmbeddingSENet(SEBasicBlock,[3, 4, 6, 3],with_variation=True).to(device)
    model = nn.DataParallel(model,device_ids=[0,1,2,3])

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    def save_model(name):
        torch.save(model.state_dict(), osp.join(args.save_path, name + '.pth'))

    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0

    timer = Timer()

    for epoch in range(1, args.max_epoch + 1):
        lr_scheduler.step()

        model.train()

        tl = Averager()
        ta = Averager()

        for i, batch in enumerate(train_loader, 1):
            data, _ = [_.to(device) for _ in batch]
            p = args.shot * args.train_way
            data_shot, data_query = data[:p], data[p:]

            proto1,proto2,proto3,proto4,std_mean = model(data_shot)
            proto1 = proto1.reshape(args.shot, args.train_way, -1).mean(dim=0)
            proto2 = proto2.reshape(args.shot, args.train_way, -1).mean(dim=0)
            proto3 = proto3.reshape(args.shot, args.train_way, -1).mean(dim=0)
            proto4 = proto4.reshape(args.shot, args.train_way, -1).mean(dim=0)

            label = torch.arange(args.train_way).repeat(args.query)
            label = label.to(device)

            query1,query2,query3,query4,_ = model(data_query)

            logits_1 = euclidean_metric(query1, proto1)
            logits_2 = euclidean_metric(query2, proto2)
            logits_3 = euclidean_metric(query3, proto3)
            logits_4 = euclidean_metric(query4, proto4)
            logits = 0.3*logits_1+0.4*logits_2+0.5*logits_3+logits_4

            loss = F.cross_entropy(logits, label) - 0.05*torch.mean(std_mean)
            acc = count_acc(logits, label)
            print('epoch {}, train {}/{}, loss={:.4f} acc={:.4f}'
                  .format(epoch, i, len(train_loader), loss.item(), acc))

            tl.add(loss.item())
            ta.add(acc)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        save_model('epoch-last')

        if epoch % args.save_epoch == 0:
            save_model('epoch-{}'.format(epoch))

        print('ETA:{}/{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch)))

