from __future__ import print_function
import argparse
import os
import time
import torch
import torch.optim as optim
from caffenet import CaffeNet
from prototxt import parse_solver

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
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
snapshot_prefix = solver['snapshot_prefix']

torch.manual_seed(int(time.time()))
if args.gpu:
    torch.cuda.manual_seed(int(time.time()))

model = CaffeNet(protofile)
print(model)

if args.gpu:
    device_ids = args.gpu.split(',')
    device_ids = [int(i) for i in device_ids]
    if len(device_ids > 1):
        model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
    else:
        model = model.cuda()

optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=momentum, weight_decay=weight_decay)

if args.weights:
    state = torch.load(args.weights)
    start_epoch = state['batch']+1
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('loaded state %s' % (args.weights))

model.set_phase('TRAIN')
for batch in range(max_iter):
    if (batch+1) % test_interval == 0:
        model.set_phase('TEST')
        average_accuracy = 0.0
        for i in range(test_iter):
            accuracy = model()['accuracy']
            average_accuracy += accuracy.data[0]
        average_accuracy /= test_iter
        print('average accuracy: %f' % average_accuracy)
        model.train()
        model.set_phase('TRAIN')
    else:
        optimizer.zero_grad()
        model()
        loss = model.get_loss()
        loss.backward()
        optimizer.step()

    savename = '%s_batch%08d.pth' % (snapshot_prefix, batch)
    print('save state %s' % (savename))
    state = {'batch': batch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()}
    torch.save(state, savename)
