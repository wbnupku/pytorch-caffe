#python train.py --solver SFD/solver.prototxt --gpu 0,1,2,3
from __future__ import print_function
import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from caffenet import CaffeNet, ParallelCaffeNet
from prototxt import parse_solver
import caffe

class CaffeDataLoader:
    def __init__(self, protofile):
        caffe.set_mode_cpu()
        self.net = caffe.Net(protofile, 'aaa', caffe.TRAIN)
        self.ngpus = 1

    def set_ngpus(self, ngpus):
        self.ngpus = ngpus

    def next(self):
        output = self.net.forward()
        data = torch.from_numpy(self.net.blobs['data'].data)
        label = torch.from_numpy(self.net.blobs['label'].data)
        if self.ngpus > 1:
            batch_size = data.size(0)
            nlabels = label.size(2)
            label = label.expand(self.ngpus, 1, nlabels, 8).contiguous()
            sub_batch = batch_size/self.ngpus
            for i in range(self.ngpus):
                sub_label = label[i,0,:, 0]
                sub_label[sub_label > (i+1)*sub_batch] = -1
                sub_label[sub_label < i*sub_batch] = -1
                sub_label = sub_label - sub_batch * i
                label[i,0,:, 0] = sub_label

        data = Variable(data.cuda())
        label = Variable(label.cuda())
        return data, label

def adjust_learning_rate(optimizer, batch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = base_lr
    for i in range(len(stepvalues)):
        if batch >= stepvalues[i]:
            lr = lr * gamma
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Train Caffe Example')
parser.add_argument('--gpu', type=str, help='gpu ids e.g "0,1,2,3"')
parser.add_argument('--solver', type=str, help='the solver prototxt')
parser.add_argument('--data_protofile', type=str, help='the data prototxt')
parser.add_argument('--model', type=str, help='the network definition prototxt')
parser.add_argument('--snapshot', type=str, help='the snapshot solver state to resume training')
parser.add_argument('--weights', type=str, help='the pretrained weight')
parser.add_argument('--lr', type=float, help='base learning rate')
args = parser.parse_args()
print(args)

solver        = parse_solver(args.solver)
protofile     = solver['train_net']
base_lr       = float(solver['base_lr'])
gamma         = float(solver['gamma'])
momentum      = float(solver['momentum'])
weight_decay  = float(solver['weight_decay'])
display       = int(solver['display'])
test_iter     = 0
max_iter      = int(solver['max_iter'])
test_interval = 99999999
snapshot      = int(solver['snapshot'])
snapshot_prefix = solver['snapshot_prefix']
stepvalues    = solver['stepvalue']
stepvalues    = [int(item) for item in stepvalues]

if args.lr != None:
    base_lr = args.lr

#torch.manual_seed(int(time.time()))
#if args.gpu:
#    torch.cuda.manual_seed(int(time.time()))

net = CaffeNet(protofile, omit_data_layer=True)
data_loader = CaffeDataLoader(args.data_protofile)
if args.weights:
    net.load_weights(args.weights)
net.set_verbose(False)
net.set_train_outputs('mbox_loss')
#net.set_eval_outputs('loss', 'accuracy')

if args.gpu:
    device_ids = args.gpu.split(',')
    device_ids = [int(i) for i in device_ids]
    print('device_ids', device_ids)
    if len(device_ids) > 1:
        print('---- Multi GPUs ----')
        #net = ParallelCaffeNet(net.cuda(), device_ids=device_ids)
        net = nn.DataParallel(net.cuda(), device_ids=device_ids)
        data_loader.set_ngpus(len(device_ids))
        #net = nn.DataParallel(net.cuda())
    else:
        print('---- Single GPU ----')
        net.cuda()

print(net)

optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=momentum, weight_decay=weight_decay)

if args.snapshot:
    state = torch.load(args.snapshot)
    start_epoch = state['batch']+1
    net.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('loaded state %s' % (args.snapshot))

net.train()
lr = adjust_learning_rate(optimizer, 0)
print('[0] init_lr = %f' % lr)
for batch in range(max_iter):
    if batch in stepvalues:
        lr = adjust_learning_rate(optimizer, batch)
        print('[%d] lr = %f' % (batch, lr))

    if (batch+1) % test_interval == 0:
        net.eval()
        average_accuracy = 0.0
        average_loss = 0.0
        #data_loader.set_phase_test()
        for i in range(test_iter):
            data, label = data_loader.next()
            loss, accuracy = net(data, label)
            average_accuracy += accuracy.data.mean()
            average_loss += loss.data.mean()
        average_accuracy /= test_iter
        average_loss /= test_iter
        print('[%d]  test loss: %f\ttest accuracy: %f' % (batch+1, average_loss, average_accuracy))
        net.train()
    else:
        #data_loader.set_phase_train()
        data, label = data_loader.next()
        optimizer.zero_grad()
        loss = net(data, label).mean()
        loss.backward()
        optimizer.step()
        if (batch+1) % display == 0:
            print('[%d] train loss: %f' % (batch+1, loss.data[0]))
        

    if (batch+1) % snapshot == 0:
        savename = '%s_batch%08d.pth' % (snapshot_prefix, batch+1)
        print('save state %s' % (savename))
        state = {'batch': batch+1,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict()}
        torch.save(state, savename)
