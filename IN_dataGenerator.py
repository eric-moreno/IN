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
import tqdm
sys.path.insert(0, '/nfshome/jduarte/DL4Jets/mpi_learn/mpi_learn/train')
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

save_path = '/bigdata/shared/BumbleB/convert_20181121_ak8_80x_deepDoubleB_db_pf_cpf_sv_dl4jets_test/'
test_2_arrays = []
test_3_arrays = []
target_test_arrays = []

for test_file in sorted(glob.glob(save_path + 'test_*_features_2.npy')):
    test_2_arrays.append(np.load(test_file))
test_2 = np.concatenate(test_2_arrays)

for test_file in sorted(glob.glob(save_path + 'test_*_features_3.npy')):
    test_3_arrays.append(np.load(test_file))
test_3 = np.concatenate(test_3_arrays)

for test_file in sorted(glob.glob(save_path + 'test_*_truth_0.npy')):
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


#Convert two sets into two branch with one set in both and one set in only one (Use for this file)
#training = training_2
test = test_2
params = params_2
#training_sv = training_3
test_sv = test_3
params_sv = params_3
N = test.shape[2]
N_sv = test_sv.shape[2]

from data import H5Data
files = glob.glob("/bigdata/shared/BumbleB/convert_20181121_ak8_80x_deepDoubleB_db_pf_cpf_sv_dl4jets_train_val/data_*.h5")
files_val = files[:5] # take first 5 for validation
files_train = files[5:] # take rest for training

label = 'new'
outdir = sys.argv[1]
os.system('mkdir -p %s'%outdir)

batch_size = 128
data_train = H5Data(batch_size = batch_size,
                    cache = None,
                    preloading=0,
                    features_name='training_subgroup', 
                    labels_name='target_subgroup')
data_train.set_file_names(files_train)
data_val = H5Data(batch_size = batch_size,
                    cache = None,
                    preloading=0,
                    features_name='training_subgroup', 
                    labels_name='target_subgroup')
data_val.set_file_names(files_train)
n_val =data_val.count_data()
n_train=data_train.count_data()
print(n_val)
print(n_train)

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


from gnn import GraphNet
n_targets = target_test.shape[1]
gnn = GraphNet(N, n_targets, len(params), 15, N_sv, len(params_sv), vv_branch=int(sys.argv[2]))

# pre load best model
#gnn.load_state_dict(torch.load('out/gnn_new_best.pth'))

#Test Set
batch_size =128
n_epochs = 200
    
loss = nn.CrossEntropyLoss(reduction='mean')
optimizer = optim.Adam(gnn.parameters(), lr = 0.0001)
loss_vals_training = np.zeros(n_epochs)
loss_std_training = np.zeros(n_epochs)
loss_vals_validation = np.zeros(n_epochs)
loss_std_validation = np.zeros(n_epochs)

acc_vals_training = np.zeros(n_epochs)
acc_vals_validation = np.zeros(n_epochs)
acc_std_training = np.zeros(n_epochs)
acc_std_validation = np.zeros(n_epochs)

final_epoch = 0
l_val_best = 99999

from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
softmax = torch.nn.Softmax(dim=1)

for m in range(n_epochs):
    print("Epoch %s\n" % m)
    #torch.cuda.empty_cache()
    final_epoch = m
    lst = []
    loss_val = []
    loss_training = []
    correct = []
    
    for sub_X,sub_Y in tqdm.tqdm(data_train.generate_data(),total=n_train/batch_size):
        training = sub_X[3]
        training_sv = sub_X[4]
        target = sub_Y[0]
        trainingv = (torch.FloatTensor(training)).cuda()
        trainingv_sv = (torch.FloatTensor(training_sv)).cuda()
        targetv = (torch.from_numpy(np.argmax(target, axis = 1)).long()).cuda()
        
        optimizer.zero_grad()
        out = gnn(trainingv.cuda(), trainingv_sv.cuda())
        l = loss(out, targetv.cuda())
        loss_training.append(l.item())
        l.backward()
        optimizer.step()
        loss_string = "Loss: %s" % "{0:.5f}".format(l.item())
        del trainingv, trainingv_sv, targetv
        
    for sub_X,sub_Y in tqdm.tqdm(data_val.generate_data(),total=n_val/batch_size):
        training = sub_X[3]
        training_sv = sub_X[4]
        target = sub_Y[0]
        trainingv = (torch.FloatTensor(training)).cuda()
        trainingv_sv = (torch.FloatTensor(training_sv)).cuda()
        targetv = (torch.from_numpy(np.argmax(target, axis = 1)).long()).cuda()

        out = gnn(trainingv.cuda(), trainingv_sv.cuda())
        lst.append(softmax(out).cpu().data.numpy())
        l_val = loss(out, targetv.cuda())
        loss_val.append(l_val.item())

        targetv_cpu = targetv.cpu().data.numpy()
        
        correct.append(target)
        del trainingv, trainingv_sv, targetv
        
    l_val = np.mean(np.array(loss_val))
    
    predicted = np.concatenate(lst) #(torch.FloatTensor(np.concatenate(lst))).to(device)
    print('\nValidation Loss: ', l_val)

    l_training = np.mean(np.array(loss_training))
    print('Training Loss: ', l_training)
    val_targetv = np.concatenate(correct) #torch.FloatTensor(np.array(correct)).cuda()
    
    torch.save(gnn.state_dict(), '%s/gnn_%s_last.pth'%(outdir,label))
    if l_val < l_val_best:
        print("new best model")
        l_val_best = l_val
        torch.save(gnn.state_dict(), '%s/gnn_%s_best.pth'%(outdir,label))
        
    
    print(val_targetv.shape, predicted.shape)
    print(val_targetv, predicted)
    acc_vals_validation[m] = accuracy_score(val_targetv[:,0],predicted[:,0]>0.5)
    print("accuracy", acc_vals_validation[m])
    loss_vals_training[m] = l_training
    loss_vals_validation[m] = l_val
    loss_std_validation[m] = np.std(np.array(loss_val))
    loss_std_training[m] = np.std(np.array(loss_training))
    if all(loss_vals_validation[max(0, m - 5):m] > min(np.append(loss_vals_validation[0:max(0, m - 5)], 200))) and m > 5:
        print('Early Stopping...')
        print(loss_vals_training, '\n', np.diff(loss_vals_training))
        break
    print()

gnn.load_state_dict(torch.load('%s/gnn_%s_best.pth'%(outdir,label)))

acc_vals_validation = acc_vals_validation[:(final_epoch)]
loss_vals_training = loss_vals_training[:(final_epoch)]
loss_vals_validation = loss_vals_validation[:(final_epoch)]
loss_std_validation = loss_std_validation[:(final_epoch)]
loss_std_training = loss_std_training[:(final_epoch)]
np.save('%s/acc_vals_validation_%s.npy'%(outdir,label),acc_vals_validation)
np.save('%s/loss_vals_training_%s.npy'%(outdir,label),loss_vals_training)
np.save('%s/loss_vals_validation_%s.npy'%(outdir,label),loss_vals_validation)
np.save('%s/loss_std_validation_%s.npy'%(outdir,label),loss_std_validation)
np.save('%s/loss_std_training_%s.npy'%(outdir,label),loss_std_training)


