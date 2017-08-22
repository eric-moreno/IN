from __future__ import print_function
import GraphNet
import argparse
import torch
import torch.nn as nn
from torch.autograd.variable import *
import torch.optim as optim
import ast
import numpy as np
import mpi_util
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
parser = argparse.ArgumentParser(description='Get GNN parameters or saved state.')
no_file = "NOFILE"

parser.add_argument('--args',
                    '-a',
                    dest='args',
                    #const=[path + 'np/traininght.npy', path + 'np/targetht.npy', 5, 6, 10, 10, 10, 5, 5, 5],
                    #default=[5, 6, 10, 10, 10, 5, 5, 5],
                    action='store',
                    nargs=8,
                    type=int,
                    help='-a De, Do, hiddenr1, hiddeno1, hiddenc1, hiddenr2, hiddeno2, hiddenc2')

parser.add_argument('--path',
                    '-p',
                    dest ='path',
                    action = 'store',
                    default = ['/bigdata/shared/HepSIM/'],
                    type=str,
                    nargs = 1)

parser.add_argument('--epoch',
                    '-e',
                    dest ='epoch',
                    action='store',
                    default = [1],
                    type=int,
                    nargs = 1)

parser.add_argument('--test',
                    '-t',
                    dest ='test',
                    action='store',
                    default = [0],
                    const = [1],
                    type=int,
                    nargs = '?')
   
def write_checkpoint(out, file_name_dict, gnn, optimizer):
    args = [gnn.N,
            gnn.n_targets,
            gnn.P,
            gnn.De,
            gnn.Do,
            gnn.hiddenr1,
            gnn.hiddeno1,
            gnn.hiddenc1,
            gnn.hiddenr2,
            gnn.hiddeno2,
            gnn.hiddenc2]
    outf = open(out + '_tmp', 'w')
    outf.write(str(file_name_dict) + '\n')
    outf.write(str(args) + '\n')
    outf.write(str(list(val_acc_vals)) + '\n')
    outf.write(str(done))
    outf.close()
    torch.save(gnn.state_dict(), file_name_dict['gnn'] + '_tmp')
    torch.save(optimizer.state_dict(), file_name_dict['optimizer'] + '_tmp')
    os.rename(out + '_tmp', out)
    os.rename(file_name_dict['gnn'] + '_tmp', file_name_dict['gnn'])
    os.rename(file_name_dict['optimizer'] + '_tmp', file_name_dict['optimizer'])
    
    
def get_training(path):
    training = torch.load(path + 'training.torch')
    target = torch.load(path + 'target.torch')
    val = torch.load(path + 'val.torch')
    val_target = torch.load(path + 'val_target.torch')
    return training, target, val, val_target

def read_checkpoint(checkpoint):
    inf = open(checkpoint, 'r')
    file_dict, args, val_acc_vals, done_str = [i.strip() for i in inf.readlines()]
    inf.close()
    file_name_dict = ast.literal_eval(file_dict)
    val_acc_vals = np.array(ast.literal_eval(val_acc_vals))
    N, n_targets, P, De, Do, hr1, ho1, hc1, hr2, ho2, hc2 = ast.literal_eval(args)
    training, target, val, val_target = get_training(file_name_dict['training_path'])
    gnn = GraphNet.GraphNet(N, n_targets, list(range(P)), De, Do, hr1, ho1, hc1, hr2, ho2, hc2)
    gnn.load_state_dict(torch.load(file_name_dict['gnn']))
    optimizer = optim.Adam(gnn.parameters())
    optimizer.load_state_dict(torch.load(file_name_dict['optimizer']))
    return gnn, optimizer, training, target, val, val_target, file_name_dict, val_acc_vals, done_str

def accuracy(predict, target):
    _, p_vals = torch.max(predict, 1)
    r = torch.sum(target == p_vals.squeeze(1)).data.numpy()[0]
    t = target.size()[0]
    return r * 1.0 / t

def stats(predict, target):
    _, p_vals = torch.max(predict, 1)
    t = target.cpu().data.numpy()
    p_vals = p_vals.squeeze(1).data.numpy()
    vals = np.unique(t)
    for i in vals:
        ind = np.where(t == i)
        pv = p_vals[ind]
        correct = sum(pv == t[ind])
        print("  Target %s: %s/%s = %s%%" % (i, correct, len(pv), correct * 100.0/len(pv)))
    print("Overall: %s/%s = %s%%" % (sum(p_vals == t), len(t), sum(p_vals == t) * 100.0/len(t)))
    return sum(p_vals == t) * 100.0/len(t)

def get_path(data, path):
    return path + 'checkpoints/' + '-'.join([str(i) for i in data]) + '/'

def early_stopping(acc, patience = 3):
    if len(acc) < 5:
        return False
    if np.argmax(acc) <= len(acc) - patience:
        return True
    return False

def write_test_value(path, args):
    out = open(path, 'w')
    out.write('line 1\n')
    out.write('line 2\n')
    out.write(str([sum(args)]) + '\n')
    out.write('line 3\n')
    out.close()

def train_epoch(trainingv, targetv, valv, val_targetv):
    for j in range(0, trainingv.size()[0], batch_size):
        optimizer.zero_grad()
        out = gnn(trainingv[j:j + batch_size].cuda())
        l = loss(out, targetv[j:j + batch_size].cuda())
        l.backward()
        optimizer.step()
        loss_string = "Loss: %s" % "{0:.5f}".format(l.cpu().data.numpy()[0])
        mpi_util.printProgressBar(j + batch_size, trainingv.size()[0], 
                              prefix = "%s [%s/%s] " % (loss_string, 
                                                        j + batch_size, 
                                                        trainingv.size()[0]),
                                                        length = 20)
    lst = []
    for j in torch.split(valv, 100):
        a = gnn(j.cuda()).cpu().data.numpy()
        lst.append(a)
    predicted = Variable(torch.FloatTensor(np.concatenate(lst)))
    a = stats(predicted, val_targetv)
    return a

done = False
args = parser.parse_args()
test = args.test[0]
graph_args = args.args
path = args.path[0]
n_epochs = args.epoch[0]
arg_dir = get_path(graph_args, path)
checkpoint = arg_dir + 'checkpoint.txt'
best_checkpoint = arg_dir + 'best_checkpoint.txt'
file_name_dict = {i: arg_dir + i + '.torch' for i in ['gnn',
                                                      'optimizer']}
best_name_dict = {i: arg_dir + 'best_' + i + '.torch' for i in file_name_dict.keys()}
file_name_dict['training_path'] = path + 'training/'
if test == 1:
    if not os.path.isdir(arg_dir):
        os.makedirs(arg_dir)
    write_test_value(best_checkpoint, graph_args)
else:    
    batch_size = 500
    if os.path.exists(checkpoint):
        print("Resuming from checkpoint located at %s" % checkpoint)
        gnn, optimizer, training, target, val, val_target, \
        file_name_dict, val_acc_vals, done_str = read_checkpoint(checkpoint)
        done = ast.literal_eval(done_str)
        if done:
            print ("Already finished, not sure why you asked me to do this again.")
        else:
            loss = nn.CrossEntropyLoss()
            for i in range(n_epochs):
                print("Epoch %s" % i)
                val_acc_vals = np.append(val_acc_vals, train_epoch(trainingv,
                                                                   targetv,
                                                                   valv, 
                                                                   val_targetv))
                if (val_acc_vals[-1] == min(val_acc_vals)):
                    write_checkpoint(best_checkpoint, best_name_dict, gnn, optimizer)
                if early_stopping(val_acc_vals):
                    done = True
                    break
                print
                write_checkpoint(checkpoint, file_name_dict, gnn, optimizer)
    else:
        print("No checkpoint at: %s\n Creating it." %checkpoint)
        if not os.path.isdir(arg_dir):
            os.makedirs(arg_dir)
        val_acc_vals = np.array([])
        De, Do, hr1, ho1, hc1, hr2, ho2, hc2 = graph_args
        trainingv, targetv, valv, val_targetv = get_training(file_name_dict['training_path'])
        N = int(trainingv.size()[2])
        P = int(trainingv.size()[1])
        n_targets = int(max(targetv.data.numpy())) + 1
        gnn = GraphNet.GraphNet(N, n_targets, list(range(P)), De, Do, hr1, ho1, hc1, hr2, ho2, hc2)
        optimizer = optim.Adam(gnn.parameters())
        loss = nn.CrossEntropyLoss()
        for i in range(n_epochs):
            print("Epoch %s" % i)
            val_acc_vals = np.append(val_acc_vals, train_epoch(trainingv,
                                                               targetv,
                                                               valv, 
                                                               val_targetv))
            if (val_acc_vals[-1] == min(val_acc_vals)):
                write_checkpoint(best_checkpoint, best_name_dict, gnn, optimizer)
            if early_stopping(val_acc_vals):
                done = True
                break
            print
            write_checkpoint(checkpoint, file_name_dict, gnn, optimizer)
