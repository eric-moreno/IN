from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd.variable import *
import torch.optim as optim
import os
import numpy as np
import pandas as pd
import util
import setGPU
import glob
import sys
sys.path.insert(0, '/nfshome/jduarte/DL4Jets/mpi_learn/mpi_learn/train')
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

save_path = '/bigdata/shared/BumbleB/convert_20181121_ak8_80x_deepDoubleB_db_pf_cpf_sv_dl4jets_test/'
test_2_arrays = []
test_3_arrays = []
target_test_arrays = []

for test_file in sorted(glob.glob(save_path + 'test_*_features_2.npy')):
    print(test_file)
    test_2_arrays.append(np.load(test_file))
test_2 = np.concatenate(test_2_arrays)

for test_file in sorted(glob.glob(save_path + 'test_*_features_3.npy')):
    print(test_file)
    test_3_arrays.append(np.load(test_file))
test_3 = np.concatenate(test_3_arrays)

for test_file in sorted(glob.glob(save_path + 'test_*_truth_0.npy')):
    print(test_file)
    target_test_arrays.append(np.load(test_file))
target_test = np.concatenate(target_test_arrays)

del test_2_arrays
del test_3_arrays
del target_test_arrays
test_2 = np.swapaxes(test_2, 1, 2)
test_3 = np.swapaxes(test_3, 1, 2)
print(test_2.shape)
print(test_3.shape)
print(target_test.shape)

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

from data import H5Data
files = glob.glob("/bigdata/shared/BumbleB/convert_20181121_ak8_80x_deepDoubleB_db_pf_cpf_sv_dl4jets_train_val/data_*.h5")

data = H5Data(batch_size = 100000,
              cache = None,
              preloading=0,
              features_name='training_subgroup', 
              labels_name='target_subgroup')
data.set_file_names(files)

#Convert two sets into two branch with one set in both and one set in only one (Use for this file)
#training = training_2
test = test_2
params = params_2
#training_sv = training_3
test_sv = test_3
params_sv = params_3
N = test.shape[2]

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

NBINS = 40 # number of bins for loss function
MMAX = 200. # max value
MMIN = 40. # min value
LAMBDA = 0.30 # lambda for penalty

def loss_kldiv(y_in,x):
    """
    mass sculpting penlaty term using kullback_leibler_divergence
    y_in: truth [h, y]
    x: predicted NN output for y
    h: the truth mass histogram vector "one-hot encoded" (length NBINS=40)
    y: the truth categorical labels  "one-hot encoded" (length NClasses=2)
    """
    h = y_in[:,0:NBINS]
    y = y_in[:,NBINS:NBINS+2]
    
    # build mass histogram for true q events weighted by q, b prob
    h_alltag_q = K.dot(K.transpose(h), K.dot(tf.diag(y[:,0]),x))
    # build mass histogram for true b events weighted by q, b prob
    h_alltag_b = K.dot(K.transpose(h), K.dot(tf.diag(y[:,1]),x))
    
    # select mass histogram for true q events weighted by q prob; normalize
    h_qtag_q = h_alltag_q[:,0]
    h_qtag_q = h_qtag_q / K.sum(h_qtag_q,axis=0)
    # select mass histogram for true q events weighted by b prob; normalize
    h_btag_q = h_alltag_q[:,1]
    h_btag_q = h_btag_q / K.sum(h_btag_q,axis=0)
    # select mass histogram for true b events weighted by q prob; normalize        
    h_qtag_b = h_alltag_b[:,0]
    h_qtag_b = h_qtag_b / K.sum(h_qtag_b,axis=0)
    # select mass histogram for true b events weighted by b prob; normalize        
    h_btag_b = h_alltag_b[:,1]
    h_btag_b = h_btag_b / K.sum(h_btag_b,axis=0)

    # compute KL divergence between true q events weighted by b vs q prob (symmetrize?)
    # compute KL divergence between true b events weighted by b vs q prob (symmetrize?)
    return categorical_crossentropy(y, x) +         LAMBDA*kullback_leibler_divergence(h_btag_q, h_qtag_q) +         LAMBDA*kullback_leibler_divergence(h_btag_b, h_qtag_b)         


import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sns
sns.set()
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

import itertools
from sklearn import utils
use_cuda = True
device = torch.device("cuda" if use_cuda else torch.device("cpu"))

class GraphNet(nn.Module):
    def __init__(self, n_constituents, n_targets, params, hidden):
        super(GraphNet, self).__init__()
        self.hidden = int(hidden)
        self.P = len(params)
        self.N = n_constituents
        self.S = test_sv.shape[1]
        self.Nv = test_sv.shape[2]
        self.Nr = self.N * (self.N - 1)
        self.Nt = self.N * self.Nv
        self.Dr = 0
        self.De = 5
        self.Dx = 0
        self.Do = 6
        self.n_targets = n_targets
        self.assign_matrices()
        self.assign_matrices_SV()
        #self.switch = switch
        
        self.Ra = torch.ones(self.Dr, self.Nr)
        self.fr1 = nn.Linear(2 * self.P + self.Dr, self.hidden).cuda()
        self.fr1_sv = nn.Linear(self.S + self.P + self.Dr, self.hidden).cuda()
        self.fr2 = nn.Linear(self.hidden, int(self.hidden/2)).cuda()
        self.fr2_sv = nn.Linear(self.hidden, int(self.hidden/2)).cuda()
        self.fr3 = nn.Linear(int(self.hidden/2), self.De).cuda()
        self.fr3_sv = nn.Linear(int(self.hidden/2), self.De).cuda()
        self.fo1 = nn.Linear(self.P + self.Dx + (2 * self.De), self.hidden).cuda()
        self.fo2 = nn.Linear(self.hidden, int(self.hidden/2)).cuda()
        self.fo3 = nn.Linear(int(self.hidden/2), self.Do).cuda()
        self.fc1 = nn.Linear(self.Do * self.N, self.hidden).cuda()
        self.fc2 = nn.Linear(self.hidden, int(self.hidden/2)).cuda()
        self.fc3 = nn.Linear(int(self.hidden/2), self.n_targets).cuda()
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
        self.Rk = torch.zeros(self.N, self.Nt)
        self.Rv = torch.zeros(self.Nv, self.Nt)
        receiver_sender_list = [i for i in itertools.product(range(self.N), range(self.Nv))]
        for i, (k, v) in enumerate(receiver_sender_list):
            self.Rk[k, i] = 1
            self.Rv[v, i] = 1
        self.Rk = (self.Rk).cuda()
        self.Rv = (self.Rv).cuda()
        
    def forward(self, x, y):
        ###PF Candidate - PF Candidate###
        print(x.shape, self.Rr.shape)
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
        print('E',E.shape)
        print('Rr',self.Rr.shape)
        print('Ebar',Ebar.shape)
        del E
        
        ####Secondary Vertex - PF Candidate### 
        Ork = self.tmul(x, self.Rk)
        Orv = self.tmul(y, self.Rv)
        B = torch.cat([Ork, Orv], 1)
        ### First MLP ###
        B = torch.transpose(B, 1, 2).contiguous()
        B = nn.functional.relu(self.fr1_sv(B.view(-1, self.S + self.P + self.Dr)))
        B = nn.functional.relu(self.fr2_sv(B))
        E = nn.functional.relu(self.fr3_sv(B).view(-1, self.Nt, self.De))
        del B
        E = torch.transpose(E, 1, 2).contiguous()
        print('E',E.shape)
        print('Rk',self.Rk.shape)
        Ebar_sv = self.tmul(E, torch.transpose(self.Rk, 0, 1).contiguous())
        print('Ebar_sv',Ebar_sv.shape)
        del E

        ####Final output matrix###
        C = torch.cat([x, Ebar, Ebar_sv], 1)
        del Ebar
        #C = torch.cat([C, Ebar_sv], 1)
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

n_targets = test.shape[1]
gnn = GraphNet(N, n_targets, params, 15)
#gnn.load_state_dict(torch.load('gnn_SV_tracks_0.4.0.torch_dataGenerator'))

def get_sample(training1, training2, target, choice):
    target_vals = np.argmax(target, axis = 1)
    ind, = np.where(target_vals == choice)
    chosen_ind = np.random.choice(ind, 200000)
    return training1[chosen_ind], training2[chosen_ind], target[chosen_ind]

def get_sample_train(training1, training2, target, choice):
    target_vals = np.argmax(target, axis = 1)
    ind, = np.where(target_vals == choice)
    chosen_ind = ind
    #chosen_ind = np.random.choice(ind, 200000)
    return training1[chosen_ind], training2[chosen_ind], target[chosen_ind]

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

        #print(sub_X, sub_Y)
        #print(sub_X[0].shape,sub_X[1].shape,sub_X[2].shape,sub_X[3].shape,sub_X[4].shape)
        #print(sub_Y[0].shape)
        training = sub_X[3]
        training_sv = sub_X[4]
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
        training = sub_X[1]
        training_sv = sub_X[2]
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
    
    torch.save(gnn.state_dict(), 'gnn_SV_tracks_0.4.0.torch_dataGenerator_3')
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


# In[ ]:


del trainingv, training_sv, targetv, valv, valv_sv, val_targetv


# In[ ]:


del testv, testv_sv, targetv_test


# In[ ]:


torch.save(gnn.state_dict(), 'gnn_SV_tracks_0.4.0.torch_dataGenerator')


# In[ ]:


generate_control_plots()


# In[ ]:


from IPython.display import Image, display
path = '/nfshome/emoreno/IN/img/n-h-hb/'
for xlabel in params:
    display(Image(filename=path + xlabel + '-actual.png'))
    display(Image(filename=path + xlabel + '-predicted.png'))


# In[ ]:


# Generate Loss Plot
loss_vals_training = loss_vals_training[:(final_epoch)] 
loss_vals_validation = loss_vals_validation[:(final_epoch)] 
loss_validation_std = loss_validation_std[:(final_epoch)] 
loss_training_std = loss_training_std[:(final_epoch)] 
epochs = np.array(range(len(loss_vals_training)))
fig = plt.figure(figsize = (12,10))
ax1 = fig.add_subplot(111)
ax1.plot(epochs, loss_vals_training, label='training')
ax1.plot(epochs, loss_vals_validation, label='validation', color = 'green')
ax1.fill_between(epochs, loss_vals_validation - loss_validation_std/2, loss_vals_validation + loss_validation_std/2, color = 'lightgreen', label = 'Validation +/- 0.5 Std')
ax1.fill_between(epochs, loss_vals_training - loss_training_std/2, loss_vals_training + loss_training_std/2, color = 'lightblue', label = 'Training +/- 0.5 Std')
plt.legend(loc='upper right')
plt.title('Loss Plot Plain IN (Data Generator)')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.savefig('Loss_SV_tracks_data_generator')
plt.show()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 10), dpi = 200)
plt.plot(acc_vals[:final_epoch])
sns.set()
plt.title('Accuracy Plain IN (Data Generator)')
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.savefig("Accuracy_SV_tracks_dataGenerator")

plt.show()


# In[ ]:


# Generate ROC Plot
from sklearn.metrics import roc_curve, roc_auc_score
softmax = torch.nn.Softmax(dim=1)
gnn.eval()
prediction = np.array([])
out = np.array([])
for j in range(0, testv.size()[0], batch_size):
    out_test = softmax(gnn(testv[j:j + batch_size].cuda(), testv_sv[j:j + batch_size].cuda()))
    out_test = out_test.cpu().data.numpy()
   
    for i in range(len(out_test)):
        if (out_test[i][0] > out_test[i][1]): 
            prediction = np.append(prediction, out_test[i][0])
            out = np.append(out, 0)
        else: 
            prediction = np.append(prediction, out_test[i][1])
            out = np.append(out, 1)

for i in range(prediction.size): 
    if out[i] == 0: 
        prediction[i] = 1.0 - prediction[i]
        
fpr, tpr, thresholds = roc_curve(targetv_test.cpu().data.numpy(), prediction)
auc = roc_auc_score(targetv_test.cpu().data.numpy(), prediction)

fpr_DeepDoubleB = np.load('fpr_DeepDoubleB.npy')
tpr_DeepDoubleB = np.load('tpr_DeepDoubleB.npy')
dfpr_BDT = np.load('dfpr_BDT.npy')
dtpr_BDT = np.load('dtpr_BDT.npy')

plt.figure(figsize=(12,10))
lw = 2
plt.semilogy(tpr, fpr, color='darkorange',
         lw=lw, label='Plain IN (area = %0.2f)' % auc)
plt.plot(tpr_DeepDoubleB, fpr_DeepDoubleB, color='blue',
         lw=lw, label='Plain DeepDoubleB (area = 0.97)')
plt.plot(dtpr_BDT, dfpr_BDT, color='green',
         lw=lw, label='BDT (area = 0.914)' % auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([10**-3, 1])
plt.ylabel('False Positive Rate')
plt.xlabel('True Positive Rate')
plt.title('ROC Plain IN (Data Generator)')
plt.legend(loc="lower right")
plt.savefig('ROC_curve_data_generator')
plt.show()

test_full = torch.FloatTensor(np.concatenate(np.array([test])))
test_sv_full = torch.FloatTensor(np.concatenate(np.array([test_sv])))
prediction_test = np.array([])
gnn_out = np.array([])
for j in range(0, test_full.size()[0], batch_size):
    print(j)
    out_test = softmax(gnn(test_full[j:j + batch_size].cuda(), test_sv_full[j:j + batch_size].cuda()))
    out_test = out_test.cpu().data.numpy()
    for i in range(len(out_test)):
        if (out_test[i][0] > out_test[i][1]): 
            prediction_test = np.append(prediction_test, out_test[i][0])
            gnn_out = np.append(gnn_out, 0)
        else: 
            prediction_test = np.append(prediction_test, out_test[i][1])
            gnn_out = np.append(gnn_out, 1)

for i in range(prediction_test.size): 
    if gnn_out[i] == 0: 
        prediction_test[i] = 1.0 - prediction_test[i]
    
#np.save('out', out)
#np.save('prediction', prediction_test)

train_full = torch.FloatTensor(np.concatenate(np.array([train])))
train_sv_full = torch.FloatTensor(np.concatenate(np.array([train_sv])))
prediction_train = np.array([])
gnn_out = np.array([])
for j in range(0, test_full.size()[0], batch_size * 10):
    print(j)
    out_test = softmax(gnn(test_full[j:j + batch_size * 10].cuda(), test_sv_full[j:j + batch_size * 10].cuda()))
    out_test = out_test.cpu().data.numpy()
    for i in range(len(out_test)):
        if (out_test[i][0] > out_test[i][1]): 
            prediction_test = np.append(prediction_test, out_test[i][0])
            gnn_out = np.append(gnn_out, 0)
        else: 
            prediction_test = np.append(prediction_test, out_test[i][1])
            gnn_out = np.append(gnn_out, 1)

for i in range(prediction_test.size): 
    if gnn_out[i] == 0: 
        prediction_test[i] = 1.0 - prediction_test[i]
    


# In[ ]:


np.save('out', gnn_out)
np.save('prediction', prediction_test)
np.save('tpr', tpr)
np.save('fpr', fpr)
np.save('thresholds', thresholds)


# In[ ]:


torch.cuda.empty_cache()

