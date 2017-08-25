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
import os, json, time
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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

parser.add_argument('--kfold',
                    '-k',
                    dest = 'kfold',
                    type = int)

parser.add_argument('--kmax',
                    '-m',
                    dest = 'kfold_max',
                    type = int)

parser.add_argument('--test',
                    '-t',
                    dest='test',
                    action = 'store_true')

parser.add_argument('--gpu',  #Use CPU unless I receive this flag
                    '-g',
                    dest = 'gpu',
                    action = 'store_true'
                    ) 
parser.set_defaults(test = False, kfold_max = 10, gpu = False)

def write_checkpoint(out, file_name_dict, gnn, optimizer, 
                     args, val_acc_vals, done, kfold, 
                     train_time, gpu):
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
    out_dict = {'file_name_dict':file_name_dict, 
                'args': args, 
                'val_acc_vals': val_acc_vals.tolist(),
                'done': done,
                'kfold': kfold,
                'train_time': train_time,
                'gpu': gpu
               }
    json_out = open(out + '_tmp', 'w')
    json_out.write(json.dumps(out_dict))
    json_out.close()
    torch.save(gnn.state_dict(), file_name_dict['gnn'] + '_tmp')
    torch.save(optimizer.state_dict(), file_name_dict['optimizer'] + '_tmp')
    os.rename(out + '_tmp', out)
    os.rename(file_name_dict['gnn'] + '_tmp', file_name_dict['gnn'])
    os.rename(file_name_dict['optimizer'] + '_tmp', file_name_dict['optimizer'])
    
    
def get_training(path, kfold, kfold_max):
    training = torch.load(path + 'training.torch')
    chunks = list(torch.chunk(training, kfold_max))
    val = chunks[kfold]
    chunks.pop(kfold)
    training = torch.cat(chunks)
    target = torch.load(path + 'target.torch')
    chunks = list(torch.chunk(target, kfold_max))
    val_target = chunks[kfold]
    chunks.pop(kfold)
    target = torch.cat(chunks)
    return training, target, val, val_target

def read_checkpoint(checkpoint, kfold, kfold_max, gpu):
    in_json  = open(checkpoint, 'r')
    in_dict = json.load(in_json)
    file_name_dict = in_dict['file_name_dict']
    val_acc_vals = np.array(in_dict['val_acc_vals'])
    args = in_dict['args']
    done = in_dict['done']
    N, n_targets, P, De, Do, hr1, ho1, hc1, hr2, ho2, hc2 = args
    training, target, val, val_target = get_training(file_name_dict['training_path'], kfold, kfold_max)
    gnn = GraphNet.GraphNet(N, n_targets, list(range(P)), De, Do, hr1, ho1, hc1, hr2, ho2, hc2, use_gpu = gpu)
    gnn.load_state_dict(torch.load(file_name_dict['gnn']))
    optimizer = optim.Adam(gnn.parameters())
    if gpu == in_dict['gpu']:
        optimizer.load_state_dict(torch.load(file_name_dict['optimizer']))
    return gnn, optimizer, training, target, val, val_target, file_name_dict, val_acc_vals, done

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

def write_test_value(path, args, k):
    json_out = open(path, 'w')
    out_dict = {'file_name_dict':{}, 
                'args': args, 
                'val_acc_vals': [sum(args)],
                'done': False,
                'kfold': k,
                'train_time': 1,
                'gpu': 1
               }
    json_out.write(json.dumps(out_dict))
    json_out.close()

def train_epoch(trainingv, targetv, valv, val_targetv, gpu = False):
    for j in range(0, trainingv.size()[0], batch_size):
        optimizer.zero_grad()
        batch = trainingv[j:j + batch_size]
        batch_target = targetv[j:j + batch_size]
        if gpu:
            batch = batch.cuda()
            batch_target = batch_target.cuda()
        out = gnn(batch)
        l = loss(out, batch_target)
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
        if gpu:
            j = j.cuda()
            a = gnn(j).cpu().data.numpy()
        else:
            a = gnn(j).data.numpy()
        lst.append(a)
    predicted = Variable(torch.FloatTensor(np.concatenate(lst)))
    a = stats(predicted, val_targetv)
    return a

done = False
args = parser.parse_args()
test = args.test
graph_args = args.args
kfold = args.kfold
kfold_max = args.kfold_max
gpu = args.gpu
path = args.path[0]
n_epochs = args.epoch[0]
arg_dir = get_path(graph_args, path)
checkpoint = arg_dir + str(kfold) + '-checkpoint.json'
best_checkpoint = arg_dir + str(kfold) + '-best_checkpoint.json'
file_name_dict = {i: arg_dir + i + '.torch' for i in ['gnn',
                                                      'optimizer']}
best_name_dict = {i: arg_dir + 'best_' + i + '.torch' for i in file_name_dict.keys()}
file_name_dict['training_path'] = path + 'training/'
if test:
    write_test_value(best_checkpoint, graph_args, kfold)
else: 
    batch_size = 720
    if os.path.exists(checkpoint):
        print("Resuming from checkpoint located at %s" % checkpoint)
        gnn, optimizer, trainingv, targetv, valv, val_targetv, \
        file_name_dict, val_acc_vals, done = read_checkpoint(checkpoint, kfold, kfold_max, gpu)
        if done:
            print ("Already finished, not sure why you asked me to do this again.")
        else:
            loss = nn.CrossEntropyLoss()
            for i in range(n_epochs):
                print("Epoch %s" % i)
                start_time = time.time() 
                val_acc_vals = np.append(val_acc_vals, train_epoch(trainingv,
                                                                   targetv,
                                                                   valv, 
                                                                   val_targetv,
                                                                   gpu = gpu))
                end_time = time.time()
                train_time = end_time - start_time
                print("Time for epoch: %s" % train_time)
                if (val_acc_vals[-1] == max(val_acc_vals)):
                    write_checkpoint(best_checkpoint, best_name_dict, gnn, optimizer, 
                                     args, val_acc_vals, done, kfold, train_time,
                                     gpu)
                if early_stopping(val_acc_vals):
                    done = True
                    break
                print
                write_checkpoint(checkpoint, file_name_dict, gnn, optimizer, 
                                 args, val_acc_vals, done, kfold, train_time,
                                 gpu)
    else:
        print("No checkpoint at: %s\n Creating it." %checkpoint)
        val_acc_vals = np.array([])
        De, Do, hr1, ho1, hc1, hr2, ho2, hc2 = graph_args
        trainingv, targetv, valv, val_targetv = get_training(file_name_dict['training_path'], 
                                                             kfold, kfold_max)
        N = int(trainingv.size()[2])
        P = int(trainingv.size()[1])
        n_targets = int(max(targetv.data.numpy())) + 1
        gnn = GraphNet.GraphNet(N, n_targets, list(range(P)), De, Do, hr1, ho1, hc1, hr2, ho2, hc2, use_gpu = gpu)
        optimizer = optim.Adam(gnn.parameters())
        loss = nn.CrossEntropyLoss()
        for i in range(n_epochs):
            print("Epoch %s" % i)
            start_time = time.time() 
            val_acc_vals = np.append(val_acc_vals, train_epoch(trainingv,
                                                               targetv,
                                                               valv, 
                                                               val_targetv,
                                                               gpu))
            end_time = time.time()
            train_time = end_time - start_time
            print("Time for epoch: %s" % train_time)
            if val_acc_vals[-1] < 51:
                gnn = GraphNet.GraphNet(N, n_targets, list(range(P)), De, Do, hr1, ho1, hc1, hr2, ho2, hc2, use_gpu = gpu)
            if (val_acc_vals[-1] == max(val_acc_vals)):
                write_checkpoint(best_checkpoint, best_name_dict, gnn, optimizer, 
                                 args, val_acc_vals, done, kfold, train_time,
                                 gpu)
            if early_stopping(val_acc_vals):
                done = True
                break
            print
            write_checkpoint(checkpoint, file_name_dict, gnn, optimizer, 
                             args, val_acc_vals, done, kfold, train_time,
                             gpu)
