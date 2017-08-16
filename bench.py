import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sns
import torch
import torch.nn as nn
from torch.autograd.variable import *
import torch.optim as optim
import os
import numpy as np
import pandas as pd
import util
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
sns.set()

def accuracy(predict, target):
    _, p_vals = torch.max(predict, 1)
    r = torch.sum(target == p_vals.squeeze(1)).data.numpy()[0]
    t = target.size()[0]
    return r * 1.0 / t

def stats(predict, target):
    print "stats"
    return 0
    _, p_vals = torch.max(predict, 1)
    t = target.data.numpy()
    p_vals = p_vals.squeeze(1).data.numpy()
    vals = np.unique(t)
    for i in vals:
        ind = np.where(t == i)
        pv = p_vals[ind]
        correct = sum(pv == t[ind])
        print "  Target %s: %s/%s = %s%%" % (i, correct, len(pv), correct * 100.0/len(pv))
    percent = sum(p_vals == t) * 100.0/len(t)
    print "Overall: %s/%s = %s%%" % (sum(p_vals == t), len(t), percent)
    return percent

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n + 1)

def predicted_histogram(data, 
                        target, 
                        labels = None, 
                        nbins = 10, 
                        out = None,
                        xlabel = None,
                        title = None
                       ):
    """@params:
        data = n x 1 array of parameter values
        target = n x categories array of predictions
    """
    target = preprocessing.normalize(target, norm = "l1")
    if labels == None:
        labels = ["" for i in range(target.shape[1])]
    #1 decide bins
    ma = np.amax(data) * 1.0
    mi = np.amin(data)
    bins = np.linspace(mi, ma, nbins)
    bin_size = bins[1] - bins[0]
    bin_locs = np.digitize(data, bins, right = True)
    #2 set up bin x category matrix
    #  Each M(bin, category) = Sum over particles with param in bin of category
    M = np.array([np.sum(target[np.where(bin_locs == i)], axis = 0) 
                  for i in range(nbins)])
    #3 plot each category/bin
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    bars = np.array([M[:, i] for i in range(M.shape[1])])
    cmap = get_cmap(len(bars), 'viridis')
    for i in range(len(bars)):
        ax.bar(bins, bars[i], 
               bottom = sum(bars[:i]), 
               color = cmap(i), 
               label = labels[i],
               width = bin_size
              )
    ax.set_xlabel(xlabel)
    ax.set_yticks([])
    ax.set_title(title)
    ax.legend()

def generate_control_plots():
    len_params = len(params)
    path = '/bigdata/shared/HepSIM/img/1ep-'
    pred = gnn(valv)
    d_target = np.array([util.get_list_from_num(i, length = n_targets) 
                             for i in val_targetv.cpu().data.numpy()])
    p_target = pred.cpu().data.numpy()
    for i in range(len(params)):
        xlabel = params[i]
        labels = ["None", "H", "H + b"]
        data = np.mean(valv.cpu().data.numpy()[:, i, :], axis = 1)
        predicted_histogram(data, d_target, 
                            nbins = 50, labels = labels,
                            xlabel = xlabel, 
                            title = "Actual Distribution"
                           )
        plt.savefig(path + xlabel + "-actual.png", dpi = 200)
        predicted_histogram(data, p_target, 
                            nbins = 50, labels = labels,
                            xlabel = xlabel,
                            title = "Predicted Distribution"
                           )
        plt.savefig(path + xlabel + "-predicted.png", dpi = 200)
        plt.close("all")

class GraphNet(nn.Module):
    def __init__(self, n_constituents, n_targets, params, hidden):
        super(GraphNet, self).__init__()
        self.hidden = hidden
        self.P = len(params)
        self.N = n_constituents
        self.Nr = self.N ** 2
        self.Dr = 0
        self.De = 5
        self.Dx = 0
        self.Do = 6
        self.n_targets = n_targets
        self.Rr = Variable(torch.ones(self.N, self.Nr))
        self.Rs = Variable(torch.ones(self.N, self.Nr))
        self.Ra = Variable(torch.ones(self.Dr, self.Nr))
        self.fr1 = nn.Linear(2 * self.P + self.Dr, hidden).cuda()
        self.fr2 = nn.Linear(hidden, hidden/2).cuda()
        self.fr3 = nn.Linear(hidden/2, self.De).cuda()
        self.fo1 = nn.Linear(self.P + self.Dx + self.De, hidden).cuda()
        self.fo2 = nn.Linear(hidden, hidden/2).cuda()
        self.fo3 = nn.Linear(hidden/2, self.Do).cuda()
        self.fc1 = nn.Linear(self.Do * self.N, hidden).cuda()
        self.fc2 = nn.Linear(hidden, hidden/2).cuda()
        self.fc3 = nn.Linear(hidden/2, self.n_targets).cuda()
    
    def forward(self, x):
        Orr = self.tmul(x, self.Rr)
        Ors = self.tmul(x, self.Rs)
        B = torch.cat([Orr, Ors], 1)
        ### First MLP ###
        B = torch.transpose(B, 1, 2).contiguous().cuda()
        B = nn.functional.relu(self.fr1(B.view(-1, 2 * self.P + self.Dr)))
        B = nn.functional.relu(self.fr2(B))
        E = nn.functional.relu(self.fr3(B).view(-1, self.Nr, self.De))
        del B
        E = torch.transpose(E, 1, 2).contiguous().cpu()
        Ebar = self.tmul(E, torch.transpose(self.Rr, 0, 1).contiguous())
        del E
        C = torch.cat([x, Ebar], 1)
        C = torch.transpose(C, 1, 2).contiguous().cuda()
        ### Second MLP ###
        C = nn.functional.relu(self.fo1(C.view(-1, self.P + self.Dx + self.De)))
        C = nn.functional.relu(self.fo2(C))
        O = nn.functional.relu(self.fo3(C).view(-1, self.N, self.Do))
        del C
        ### Classification MLP ###
        N = nn.functional.relu(self.fc1(O.view(-1, self.Do * self.N)))
        del O
        N = nn.functional.relu(self.fc2(N))
        N = nn.functional.relu(self.fc3(N))
        return N.cpu()

    def tmul(self, x, y):  #Takes (I * J * K)(K * L) -> I * J * L 
        x_shape = x.size()
        y_shape = y.size()
        return torch.mm(x.view(-1, x_shape[2]), y).view(-1, x_shape[1], y_shape[1])
    
def get_sample(training, target, choice):
    target_vals = np.argmax(target, axis = 1)
    ind, = np.where(target_vals == choice)
    chosen_ind = np.random.choice(ind, 10000)
    return training[chosen_ind], target[chosen_ind]

N = 30
training = np.load('training.npy')
target = np.load('target.npy')
params = ['Px','Py', 'Pz', 'PT', 'E', 'D0', 'DZ', 'X', 'Y',  'Z', 'T', 'count']

n_targets = target.shape[1]
samples = [get_sample(training, target, i) for i in range(n_targets)]
trainings = [i[0] for i in samples]
targets = [i[1] for i in samples]
big_training = np.concatenate(trainings)
big_target = np.concatenate(targets)
big_training, big_target = util.shuffle_together(big_training, big_target)

val_split = 0.1
batch_size = 1000
n_epochs = 100

trainingv = Variable(torch.FloatTensor(big_training))
targetv = Variable(torch.from_numpy(np.argmax(big_target, axis = 1)).long())  
trainingv, valv = torch.split(trainingv, int(trainingv.size()[0] * (1 - val_split)))
targetv, val_targetv = torch.split(targetv, int(targetv.size()[0] * (1 - val_split)))
gnn = GraphNet(N, n_targets, params, 10)
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(gnn.parameters())
loss_vals = np.zeros(n_epochs)
acc_vals = np.zeros(n_epochs)
for i in range(n_epochs):
    print "Epoch %s" % i
    for j in range(0, trainingv.size()[0], batch_size):
        optimizer.zero_grad()
        out = gnn(trainingv[j:j + batch_size])
        l = loss(out, targetv[j:j + batch_size])
        l.backward()
        optimizer.step()
        lv = l.data.numpy()[0]
        loss_string = "Loss: %s" % "{0:.5f}".format(lv)
        util.printProgressBar(j + batch_size, trainingv.size()[0], 
                              prefix = "%s [%s/%s] " % (loss_string, 
                                                       j + batch_size, 
                                                       trainingv.size()[0]),
                              length = 20)
    pred = torch.cat([gnn(j) for j in torch.split(valv, batch_size)])
    acc_vals[i] = stats(pred, val_targetv)
    print "next"
    loss_vals[i] = lv
    print i, loss_vals
    if all(np.diff(loss_vals)[max(0, i - 5):i] > 0) and i > 5:
        print loss_vals, '\n', np.diff(loss_vals)
        break
    print
