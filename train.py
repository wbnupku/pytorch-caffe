from __future__ import print_function
import argparse
import os
import time
import torch
import torch.optim as optim
from caffenet import CaffeNet
from prototxt import parse_solver

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Train Caffe Example')
parser.add_argument('--gpu', help='gpu ids e.g "0,1,2,3"')
parser.add_argument('--solver', help='the solver prototxt')
parser.add_argument('--model', help='the network definition prototxt')
parser.add_argument('--snapshot', help='the snapshot solver state to resume training')
parser.add_argument('--weights', help='the pretrained weight')
args = parser.parse_args()

solver        = parse_solver(args.solver)
protofile     = solver['net']
base_lr       = float(solver['base_lr'])
momentum      = float(solver['momentum'])
weight_decay  = float(solver['weight_decay'])
test_iter     = int(solver['test_iter'])
max_iter      = int(solver['max_iter'])
test_interval = int(solver['test_interval'])
snapshot      = int(solver['snapshot'])
snapshot_prefix = solver['snapshot_prefix']

torch.manual_seed(int(time.time()))
if args.gpu:
    torch.cuda.manual_seed(int(time.time()))

model = CaffeNet(protofile)
model.set_verbose(False)
print(model)

net = model
if args.gpu:
    device_ids = args.gpu.split(',')
    device_ids = [int(i) for i in device_ids]
    print('device_ids', device_ids)
    if len(device_ids) > 1:
        print('--- multi gpus ---')
        net = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
    else:
        print('--- single gpu ---')
        net = model.cuda()

optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=momentum, weight_decay=weight_decay)

if args.weights:
    state = torch.load(args.weights)
    start_epoch = state['batch']+1
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('loaded state %s' % (args.weights))

net.train()
model.set_phase('TRAIN')
model.set_outputs('loss')
for batch in range(max_iter):
    if (batch+1) % test_interval == 0:
        net.eval()
        model.set_phase('TEST')
        model.set_outputs('loss', 'accuracy')
        average_accuracy = 0.0
        average_loss = 0.0
        for i in range(test_iter):
            loss, accuracy = net()
            average_accuracy += accuracy.data[0]
            average_loss += loss.data[0]
        average_accuracy /= test_iter
        average_loss /= test_iter
        print('[%d] accuracy: %f' % (batch+1, average_accuracy))
        print('[%d]     loss: %f' % (batch+1, average_loss))
        net.train()
        model.set_phase('TRAIN')
        model.set_outputs('loss')
    else:
        optimizer.zero_grad()
        loss = net()
        loss.backward()
        optimizer.step()

    if (batch+1) % snapshot == 0:
        savename = '%s_batch%08d.pth' % (snapshot_prefix, batch+1)
        print('save state %s' % (savename))
        state = {'batch': batch+1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()}
        torch.save(state, savename)
