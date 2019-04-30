#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
from torch.autograd.variable import *
import torch.optim as optim
import os
import numpy as np
import pandas as pd
import util
from __future__ import print_function
import setGPU
#os.environ['CUDA_VISIBLE_DEVICES']="6,7"
from matplotlib import rcParams
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rcParams['font.size'] = 15
rcParams['text.latex.preamble'] = [
#       r'\usepackage{siunitx}',   # i need upright \micro symbols, but you need...
#       r'\sisetup{detect-all}',   # ...this to force siunitx to actually use your fonts
       r'\usepackage{helvet}',    # set the normal font here
       r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
       r'\sansmath'               # <- tricky! -- gotta actually tell tex to use!
]  
rc('text', usetex=True)
import matplotlib.ticker as plticker


# In[ ]:


save_path = '/nfshome/emoreno/IN/data/opendata/test/'
#test_0 = np.load(save_path + 'test_features_0.npy')
#test_0 = np.swapaxes(test_0, 1, 2)
#training_0 = np.load(save_path + 'train_withSpectator_features_0.npy') #per jet constituents
#training_0 = np.swapaxes(training_0, 1, 2)
#training_2 = np.load(save_path + 'train_features_2.npy') #30 features of 60 charged particles
#training_2 = np.swapaxes(training_2, 1, 2)
#training_3 = np.load(save_path + 'train_features_3.npy') #14 features of 5 secondary vertices
#training_3 = np.swapaxes(training_3, 1, 2)
#target = np.load(save_path + 'train_truth_0.npy')
#test_0 = np.load(save_path + 'test_0.npy')
#test_0 = np.swapaxes(test_0, 1, 2)
#print(test_0.shape)
#test_1 = np.load(save_path + 'test_1.npy')
#test_1 = np.swapaxes(test_1, 1, 2)

# Load tracks dataset
test_2 = np.load(save_path + 'test_2.npy')
test_2 = np.swapaxes(test_2, 1, 2)

# Load SV dataset
test_3 = np.load(save_path + 'test_3.npy')
test_3 = np.swapaxes(test_3, 1, 2)
target_test = np.load(save_path + 'truth_0.npy')
params_0 = ['fj_jetNTracks',
          'fj_nSV',
          'fj_tau0_trackEtaRel_0',
          'fj_tau0_trackEtaRel_1',
          'fj_tau0_trackEtaRel_2',
          'fj_tau1_trackEtaRel_0',
          'fj_tau1_trackEtaRel_1',
          'fj_tau1_trackEtaRel_2',
          'fj_tau_flightDistance2dSig_0',
          'fj_tau_flightDistance2dSig_1',
          'fj_tau_vertexDeltaR_0',
          'fj_tau_vertexEnergyRatio_0',
          'fj_tau_vertexEnergyRatio_1',
          'fj_tau_vertexMass_0',
          'fj_tau_vertexMass_1',
          'fj_trackSip2dSigAboveBottom_0',
          'fj_trackSip2dSigAboveBottom_1',
          'fj_trackSip2dSigAboveCharm_0',
          'fj_trackSipdSig_0',
          'fj_trackSipdSig_0_0',
          'fj_trackSipdSig_0_1',
          'fj_trackSipdSig_1',
          'fj_trackSipdSig_1_0',
          'fj_trackSipdSig_1_1',
          'fj_trackSipdSig_2',
          'fj_trackSipdSig_3',
          'fj_z_ratio'
          ]

params_1 = ['pfcand_ptrel',
          'pfcand_erel',
          'pfcand_phirel',
          'pfcand_etarel',
          'pfcand_deltaR',
          'pfcand_puppiw',
          'pfcand_drminsv',
          'pfcand_drsubjet1',
          'pfcand_drsubjet2',
          'pfcand_hcalFrac'
         ]

params_2 = ['track_ptrel',     
          'track_erel',     
          'track_phirel',     
          'track_etarel',     
          'track_deltaR',
          'track_drminsv',     
          'track_drsubjet1',     
          'track_drsubjet2',
          'track_dz',     
          'track_dzsig',     
          'track_dxy',     
          'track_dxysig',     
          'track_normchi2',     
          'track_quality',     
          'track_dptdpt',     
          'track_detadeta',     
          'track_dphidphi',     
          'track_dxydxy',     
          'track_dzdz',     
          'track_dxydz',     
          'track_dphidxy',     
          'track_dlambdadz',     
          'trackBTag_EtaRel',     
          'trackBTag_PtRatio',     
          'trackBTag_PParRatio',     
          'trackBTag_Sip2dVal',     
          'trackBTag_Sip2dSig',     
          'trackBTag_Sip3dVal',     
          'trackBTag_Sip3dSig',     
          'trackBTag_JetDistVal'
         ]

params_3 = ['sv_ptrel',
          'sv_erel',
          'sv_phirel',
          'sv_etarel',
          'sv_deltaR',
          'sv_pt',
          'sv_mass',
          'sv_ntracks',
          'sv_normchi2',
          'sv_dxy',
          'sv_dxysig',
          'sv_d3d',
          'sv_d3dsig',
          'sv_costhetasvpv'
         ]


# In[ ]:


from data import H5Data

files = []
for i in range(52):
    files.append("/nfshome/emoreno/IN/data/opendata/train/data_" + str(i))

data = H5Data(batch_size = 100000,
               cache = None,
               preloading=0,
               features_name='training_subgroup', labels_name='target_subgroup')
data.set_file_names(files)


# In[ ]:


test = test_2
params = params_2
test_sv = test_3
params_sv = params_3
N = test.shape[2]


# In[ ]:


def accuracy(predict, target):
    _, p_vals = torch.max(predict, 1)
    r = torch.sum(target == p_vals.squeeze(1)).data.numpy()[0]
    t = target.size()[0]
    return r * 1.0 / t

def stats(predict, target):
    _, p_vals = torch.max(predict, 1)
    t = target.cpu().data.numpy()
    p_vals = p_vals.squeeze(0).cpu().data.numpy()
    vals = np.unique(t)
    for i in vals:
        ind = np.where(t == i)
        pv = p_vals[ind]
        correct = sum(pv == t[ind])
        print("  Target %s: %s/%s = %s%%" % (i, correct, len(pv), correct * 100.0/len(pv)))
    print("Overall: %s/%s = %s%%" % (sum(p_vals == t), len(t), sum(p_vals == t) * 100.0/len(t)))
    return sum(p_vals == t) * 100.0/len(t)

NBINS=40 # number of bins for loss function
MMAX = 200. # max value
MMIN = 40. # min value
LAMBDA = 0.30 # lambda for penalty

def loss_kldiv(y_in,x):
    """
    mass sculpting penlaty term usking kullback_leibler_divergence
    y_in: truth [h, y]
    x: predicted NN output for y
    h: the truth mass histogram vector "one-hot encoded" (length NBINS=40)
    y: the truth categorical labels  "one-hot encoded" (length NClasses=2)
    """
    h = y_in[:,0:NBINS]
    y = y_in[:,NBINS:NBINS+2]
    h_all = K.dot(K.transpose(h), y)
    h_all_q = h_all[:,0]
    h_all_h = h_all[:,1]
    h_all_q = h_all_q / K.sum(h_all_q,axis=0)
    h_all_h = h_all_h / K.sum(h_all_h,axis=0)
    h_btag_anti_q = K.dot(K.transpose(h), K.dot(tf.diag(y[:,0]),x))
    h_btag_anti_h = K.dot(K.transpose(h), K.dot(tf.diag(y[:,1]),x))
    h_btag_q = h_btag_anti_q[:,1]
    h_btag_q = h_btag_q / K.sum(h_btag_q,axis=0)
    h_anti_q = h_btag_anti_q[:,0]
    h_anti_q = h_anti_q / K.sum(h_anti_q,axis=0)
    h_btag_h = h_btag_anti_h[:,1]
    h_btag_h = h_btag_h / K.sum(h_btag_h,axis=0)
    h_anti_h = h_btag_anti_q[:,0]
    h_anti_h = h_anti_h / K.sum(h_anti_h,axis=0)

    return categorical_crossentropy(y, x) +         LAMBDA*kullback_leibler_divergence(h_btag_q, h_anti_q) +         LAMBDA*kullback_leibler_divergence(h_btag_h, h_anti_h)         


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from sklearn import preprocessing
#import seaborn as sns
#sns.set()
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
    #global gnn
    len_params = len(params)
    path = '/nfshome/emoreno/IN/img/n-h-hb/'
    #os.makedirs(path)
    fr = 0
    b = 1000
    pred= None
    while fr< valv.shape[0]: #beginning splitting up valv into batches because memory runs out
        print ("Predicting from",fr)
        valv_1 = valv[fr:fr+b,...]
        p = gnn(valv_1.cuda())
        valv_1.cpu()
        p = p.cpu().data
        fr +=b
        if pred is None:
            pred = p
        else:
            pred = np.append(pred,p,axis=0)
        print (pred.shape) #end 

    d_target = np.array([util.get_list_from_num(i, length = n_targets) 
                             for i in val_targetv.cpu().data.numpy()])
    p_target = pred#.cpu().data.numpy()
    for i in range(len(params)):
        xlabel = params[i]
        labels = ["None", "H", "H + b"]
        data = np.mean(valv.data.numpy()[:, i, :], axis = 1)
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
    plt.show()


# In[ ]:


import itertools
from sklearn import utils
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")

class GraphNet(nn.Module):
    def __init__(self, n_constituents, n_targets, params, hidden):
        super(GraphNet, self).__init__()
        self.hidden = hidden
        self.P = len(params)
        self.N = n_constituents
        self.S = test_sv.shape[1]
        self.Nv = test_sv.shape[2]
        self.Nr = self.N * (self.N - 1)
        self.Dr = 0
        self.De = 5
        self.Dx = 0
        self.Do = 6
        self.n_targets = n_targets
        self.assign_matrices()
        self.assign_matrices_SV()
        #self.switch = switch
        
        self.Ra = torch.ones(self.Dr, self.Nr)
        self.fr1 = nn.Linear(2 * self.P + self.Dr, hidden).cuda()
        self.fr1_sv = nn.Linear(self.S + self.P + self.Dr, hidden).cuda()
        self.fr2 = nn.Linear(hidden, hidden/2).cuda()
        self.fr3 = nn.Linear(hidden/2, self.De).cuda()
        self.fo1 = nn.Linear(self.P + self.Dx + (2 * self.De), hidden).cuda()
        self.fo2 = nn.Linear(hidden, hidden/2).cuda()
        self.fo3 = nn.Linear(hidden/2, self.Do).cuda()
        self.fc1 = nn.Linear(self.Do * self.N, hidden).cuda()
        self.fc2 = nn.Linear(hidden, hidden/2).cuda()
        self.fc3 = nn.Linear(hidden/2, self.n_targets).cuda()
        self.fc_fixed = nn.Linear(self.Do, self.n_targets).cuda()
        #self.gru = nn.GRU(input_size = self.Do, hidden_size = 20, bidirectional = False).cuda()
            
    def assign_matrices(self):
        self.Rr = torch.zeros(self.N, self.Nr)
        self.Rs = torch.zeros(self.N, self.Nr)
        receiver_sender_list = [i for i in itertools.product(range(self.N), range(self.N)) if i[0]!=i[1]]
        for i, (r, s) in enumerate(receiver_sender_list):
            self.Rr[r, i] = 1
            self.Rs[s, i] = 1
        self.Rr = (self.Rr).cuda()
        self.Rs = (self.Rs).cuda()
    
    def assign_matrices_SV(self):
        self.Rk = torch.zeros(self.N, self.Nr)
        self.Rv = torch.zeros(self.Nv, self.Nr)
        receiver_sender_list = [i for i in itertools.product(range(self.N), range(self.Nv)) if i[0]!=i[1]]
        for i, (k, v) in enumerate(receiver_sender_list):
            self.Rk[k, i] = 1
            self.Rv[v, i] = 1
        self.Rk = (self.Rk).cuda()
        self.Rv = (self.Rv).cuda()
        
    def forward(self, x, y):
        ###PF Candidate - PF Candidate###
        Orr = self.tmul(x, self.Rr)
        Ors = self.tmul(x, self.Rs)
        B = torch.cat([Orr, Ors], 1)
        ### First MLP ###
        B = torch.transpose(B, 1, 2).contiguous()
        B = nn.functional.relu(self.fr1(B.view(-1, 2 * self.P + self.Dr)))
        B = nn.functional.relu(self.fr2(B))
        E = nn.functional.relu(self.fr3(B).view(-1, self.Nr, self.De))
        del B
        E = torch.transpose(E, 1, 2).contiguous()
        Ebar = self.tmul(E, torch.transpose(self.Rr, 0, 1).contiguous())
        del E
        
        ####Secondary Vertex - PF Candidate### 
        Ork = self.tmul(x, self.Rk)
        Orv = self.tmul(y, self.Rv)
        B = torch.cat([Ork, Orv], 1)
        ### First MLP ###
        B = torch.transpose(B, 1, 2).contiguous()
        B = nn.functional.relu(self.fr1_sv(B.view(-1, self.S + self.P + self.Dr)))
        B = nn.functional.relu(self.fr2(B))
        E = nn.functional.relu(self.fr3(B).view(-1, self.Nr, self.De))
        del B
        E = torch.transpose(E, 1, 2).contiguous()
        Ebar_sv = self.tmul(E, torch.transpose(self.Rr, 0, 1).contiguous())
        del E

        ####Final output matrix###
        C = torch.cat([x, Ebar], 1)
        del Ebar
        C = torch.cat([C, Ebar_sv], 1)
        del Ebar_sv
        C = torch.transpose(C, 1, 2).contiguous()
        ### Second MLP ###
        C = nn.functional.relu(self.fo1(C.view(-1, self.P + self.Dx + (2 * self.De))))
        C = nn.functional.relu(self.fo2(C))
        O = nn.functional.relu(self.fo3(C).view(-1, self.N, self.Do))
        #Taking the mean/sum of each column
        #N = torch.mean(O, dim=1)
        N = torch.sum(O, dim=1)
        del C
        ### Classification MLP ###
        #N = nn.functional.relu(self.fc1(O.view(-1, self.Do * self.N)))
        del O
        #N = nn.functional.relu(self.fc2(N))
        #N = nn.functional.relu(self.fc3(N))
        N = nn.functional.relu(self.fc_fixed(N))
        #P = np.array(N.data.cpu().numpy())
        #N = np.zeros((128, 1, 6))
        #for i in range(batch_size):
        #    N[i] = np.array(np.split(P[i], self.Do))
        #    N[1] = [P[i]]
        #N, hn = self.gru(torch.tensor(N).cuda())
        #print((N).shape)
        return N 
            
    def tmul(self, x, y):  #Takes (I * J * K)(K * L) -> I * J * L 
        x_shape = x.size()
        y_shape = y.size()
        return torch.mm(x.view(-1, x_shape[2]), y).view(-1, x_shape[1], y_shape[1])

#n_targets = test.shape[1]
n_targets = 2
N = 60
gnn = GraphNet(N, n_targets, params, 15)
#gnn.load_state_dict(torch.load('IN_opendata_V2'))

def get_sample(training1, training2, target, choice):
    target_vals = np.argmax(target, axis = 1)
    ind, = np.where(target_vals == choice)
    chosen_ind = np.random.choice(ind, 300000)
    return training1[chosen_ind], training2[chosen_ind], target[chosen_ind]

def get_sample_train(training1, training2, target, choice):
    target_vals = np.argmax(target, axis = 1)
    ind, = np.where(target_vals == choice)
    chosen_ind = ind
    #chosen_ind = np.random.choice(ind, 200000)
    return training1[chosen_ind], training2[chosen_ind], target[chosen_ind]


# In[ ]:


#Test Set

val_split = 0.1
batch_size =128
n_epochs = 100

n_targets_test = target_test.shape[1]
samples_test = [get_sample(test, test_sv, target_test, i) for i in range(n_targets_test)]
tests = [i[0] for i in samples_test]
tests_sv = [i[1] for i in samples_test]
targets_tests = [i[2] for i in samples_test]

big_test = np.concatenate(tests)
big_test_sv = np.concatenate(tests_sv)
big_target_test = np.concatenate(targets_tests)
big_test, big_test_sv, big_target_test = utils.shuffle(big_test, big_test_sv, big_target_test)

testv = (torch.FloatTensor(big_test)).cuda()
testv_sv = (torch.FloatTensor(big_test_sv)).cuda()
targetv_test = (torch.from_numpy(np.argmax(big_target_test, axis = 1)).long()).cuda()
testv, valv_test = torch.split(testv, int(testv.size()[0] * (1 - val_split)))
testv_sv, valv_test_sv = torch.split(testv_sv, int(testv_sv.size()[0] * (1 - val_split)))
targetv_test, val_targetv_test = torch.split(targetv_test, int(targetv_test.size()[0] * (1 - val_split)))

loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(gnn.parameters(), lr = 0.0001)
loss_vals_training = np.zeros(n_epochs)
loss_validation_std = np.zeros(n_epochs)
loss_training_std = np.zeros(n_epochs)
loss_vals_validation = np.zeros(n_epochs)
acc_vals = np.zeros(n_epochs)
final_epoch = 0

for m in range(n_epochs):
    print("Epoch %s" % m)
    #torch.cuda.empty_cache()
    final_epoch = m
    lst = []
    loss_val = []
    loss_training = []
    correct = []
    
    for sub_X,sub_Y in data.generate_data():
        
        training = sub_X[2]
        training_sv = sub_X[3]
        target = sub_Y[0]
        # Training Set
        n_targets = target.shape[1]
        samples = [get_sample_train(training, training_sv, target, i) for i in range(n_targets)]
        trainings = [i[0] for i in samples]
        trainings_sv = [i[1] for i in samples]
        targets = [i[2] for i in samples]
        big_training = np.concatenate(trainings)
        big_training_sv = np.concatenate(trainings_sv)
        big_target = np.concatenate(targets)
        big_training, big_training_sv, big_target = utils.shuffle(big_training, big_training_sv, big_target)

        val_split = 0.1
        batch_size =128
        n_epochs = 100

        trainingv = (torch.FloatTensor(big_training)).cuda()
        trainingv_sv = (torch.FloatTensor(big_training_sv)).cuda()
        targetv = (torch.from_numpy(np.argmax(big_target, axis = 1)).long()).cuda()
        trainingv, valv = torch.split(trainingv, int(trainingv.size()[0] * (1 - val_split)))
        trainingv_sv, valv_sv = torch.split(trainingv_sv, int(trainingv_sv.size()[0] * (1 - val_split)))
        targetv, val_targetv = torch.split(targetv, int(targetv.size()[0] * (1 - val_split)))
        samples_random = np.random.choice(range(len(trainingv)), valv.size()[0]/100)

        for j in range(0, trainingv.size()[0], batch_size):
            optimizer.zero_grad()
            out = gnn(trainingv[j:j + batch_size].cuda(), trainingv_sv[j:j + batch_size].cuda())
            l = loss(out, targetv[j:j + batch_size].cuda())
            l.backward()
            optimizer.step()
            loss_string = "Loss: %s" % "{0:.5f}".format(l.item())
            util.printProgressBar(j + batch_size, trainingv.size()[0], 
                                  prefix = "%s [%s/%s] " % (loss_string, 
                                                           j + batch_size, 
                                                           trainingv.size()[0]),
                                  length = 20)
        
        del trainingv, training_sv, targetv, valv, valv_sv, val_targetv
        
    for sub_X,sub_Y in data.generate_data():
        training = sub_X[2]
        training_sv = sub_X[3]
        target = sub_Y[0]

        # Training Set
        n_targets = target.shape[1]
        samples = [get_sample_train(training, training_sv, target, i) for i in range(n_targets)]
        trainings = [i[0] for i in samples]
        trainings_sv = [i[1] for i in samples]
        targets = [i[2] for i in samples]
        big_training = np.concatenate(trainings)
        big_training_sv = np.concatenate(trainings_sv)
        big_target = np.concatenate(targets)
        big_training, big_training_sv, big_target = utils.shuffle(big_training, big_training_sv, big_target)

        val_split = 0.1
        batch_size =128
        n_epochs = 100

        trainingv = (torch.FloatTensor(big_training)).cuda()
        trainingv_sv = (torch.FloatTensor(big_training_sv)).cuda()
        targetv = (torch.from_numpy(np.argmax(big_target, axis = 1)).long()).cuda()
        trainingv, valv = torch.split(trainingv, int(trainingv.size()[0] * (1 - val_split)))
        trainingv_sv, valv_sv = torch.split(trainingv_sv, int(trainingv_sv.size()[0] * (1 - val_split)))
        targetv, val_targetv = torch.split(targetv, int(targetv.size()[0] * (1 - val_split)))
        samples_random = np.random.choice(range(len(trainingv)), valv.size()[0]/100)
        
        # Validation Loss

        for j in range(0, valv.size()[0], batch_size):
            out = gnn(valv[j:j + batch_size].cuda(), valv_sv[j:j + batch_size].cuda())
            lst.append(out.cpu().data.numpy())
            l_val = loss(out, val_targetv[j:j + batch_size].cuda())
            loss_val.append(l_val.item())

        val_targetv_cpu = val_targetv.cpu().data.numpy()
        for n in range(val_targetv_cpu.shape[0]):
            correct.append(val_targetv_cpu[n])

        # Training Loss

        for j in samples_random:
            out = gnn(trainingv[j:j + batch_size].cuda(), trainingv_sv[j:j + batch_size].cuda())
            l_training = loss(out, targetv[j:j + batch_size].cuda())
            loss_training.append(l_training.item())
        
        del trainingv, training_sv, targetv, valv, valv_sv, val_targetv

    l_val = np.mean(np.array(loss_val))
    predicted = (torch.FloatTensor(np.concatenate(lst))).to(device)
    print('\nValidation Loss: ', l_val)

    l_training = np.mean(np.array(loss_training))
    print('Training Loss: ', l_training)
    val_targetv = torch.FloatTensor(np.array(correct)).cuda()
    
    torch.save(gnn.state_dict(), 'IN_opendata_V2')
    acc_vals[m] = stats(predicted, val_targetv)
    loss_vals_training[m] = l_training
    loss_vals_validation[m] = l_val
    loss_validation_std[m] = np.std(np.array(loss_val))
    loss_training_std[m] = np.std(np.array(loss_training))
    if all(loss_vals_validation[max(0, m - 5):m] > min(np.append(loss_vals_validation[0:max(0, m - 5)], 200))) and m > 5:
        print('Early Stopping...')
        print(loss_vals_training, '\n', np.diff(loss_vals_training))
        break
    print

# Generate Test dataset Output
softmax = torch.nn.Softmax(dim=1)
test_full = torch.FloatTensor(np.concatenate(np.array([test])))
test_sv_full = torch.FloatTensor(np.concatenate(np.array([test_sv])))
prediction_test = np.array([])
gnn_out = np.array([])
IN_out = []
for j in range(0, test_full.size()[0], batch_size):
    print(j)
    out_test = softmax(gnn(test_full[j:j + batch_size].cuda(), test_sv_full[j:j + batch_size].cuda()))
    out_test = out_test.cpu().data.numpy()
    for i in range(len(out_test)):
        IN_out.append(out_test[i])

np.save('IN_out', np.array(IN_out))


# In[ ]:


# Generate Loss Plot

loss_vals_training = loss_vals_training[:(final_epoch)] 
loss_vals_validation = loss_vals_validation[:(final_epoch)] 
loss_validation_std = loss_validation_std[:(final_epoch)] 
loss_training_std = loss_training_std[:(final_epoch)] 

f, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(0,60)
ax.set_ylim(0.1, 0.3)
eraText=r'2016 (13 TeV)'
ax.set_xlabel(r'Epoch', ha='right', x=1.0)
ax.set_ylabel(r'Validation Loss', ha='right', y=1.0)
ax.tick_params(direction='in', axis='both', which='major', labelsize=15, length=12 )
ax.tick_params(direction='in', axis='both', which='minor' , length=6)
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')    
ax.grid(which='minor', alpha=0.5, axis='y', linestyle='dotted')
ax.grid(which='major', alpha=0.9, linestyle='dotted')
ax.annotate(eraText, xy=(47, 0.302), fontname='Helvetica', ha='left',
            bbox={'facecolor':'white', 'edgecolor':'white', 'alpha':0, 'pad':13}, annotation_clip=False)
ax.annotate('$\mathbf{CMS}$', xy=(0.01, 0.302), fontname='Helvetica', fontsize=24, fontweight='bold', ha='left',
            bbox={'facecolor':'white', 'edgecolor':'white', 'alpha':0, 'pad':13}, annotation_clip=False)
ax.annotate('$Simulation\ Preliminary$', xy=(6, 0.302), fontsize=18, fontstyle='italic', ha='left',
            annotation_clip=False)

epochs = np.array(range(len(loss_vals_training)))
ax.plot(epochs, loss_vals_training, label='training')
ax.plot(epochs, loss_vals_validation, label='validation', color = 'green')
ax.fill_between(epochs, loss_vals_validation - loss_validation_std/2, loss_vals_validation + loss_validation_std/2, color = 'lightgreen', label = 'Validation +/- 0.5 Std')
ax.fill_between(epochs, loss_vals_training - loss_training_std/2, loss_vals_training + loss_training_std/2, color = 'lightblue', label = 'Training +/- 0.5 Std')
ax.legend(loc='upper right', fontsize = 13)
#plt.title('Loss Plot Plain IN (Data Generator)', fontsize = 13)
#plt.ylabel('Loss', fontsize = 13)
#plt.xlabel('Epoch', fontsize = 13)
plt.savefig('Loss_opendata_Run2')
#plt.grid(True)
plt.show()



# Generate accuracy plot

f, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(0,60)
ax.set_ylim(90,95)
eraText=r'2016 (13 TeV)'
ax.set_xlabel(r'Epoch', ha='right', x=1.0)
ax.set_ylabel(r'Accuracy (\%)', ha='right', y=1.0)
#plt.figure(figsize=(12, 10), dpi = 200)
ax.plot(acc_vals)
ax.tick_params(direction='in', axis='both', which='major', labelsize=15, length=12 )
ax.tick_params(direction='in', axis='both', which='minor' , length=6)
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')    
ax.grid(which='minor', alpha=0.5, axis='y', linestyle='dotted')
ax.grid(which='major', alpha=0.9, linestyle='dotted')
ax.annotate(eraText, xy=(47, 95.1), fontname='Helvetica', ha='left',
            bbox={'facecolor':'white', 'edgecolor':'white', 'alpha':0, 'pad':13}, annotation_clip=False)
ax.annotate('$\mathbf{CMS}$', xy=(0.01, 95.1), fontname='Helvetica', fontsize=24, fontweight='bold', ha='left',
            bbox={'facecolor':'white', 'edgecolor':'white', 'alpha':0, 'pad':13}, annotation_clip=False)
ax.annotate('$Simulation\ Preliminary$', xy=(6, 95.1), fontsize=18, fontstyle='italic', ha='left',
            annotation_clip=False)
#sns.set()
#plt.title('Accuracy Plain IN (Data Generator)')
#plt.xlabel("Epoch")
#plt.ylabel("Accuracy (%)")
plt.savefig("Accuracy_opendata_Run2")
#plt.grid(True)
plt.show()