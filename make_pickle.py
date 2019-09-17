import numpy as np
import glob
import pandas as pd
import sys
import imp
try:
    imp.find_module('setGPU')
    import setGPU
except ImportError:
    pass    
import argparse
import os

if os.path.isdir('/bigdata/shared/BumbleB'):
    save_path = '/bigdata/shared/BumbleB/convert_20181121_ak8_80x_deepDoubleB_db_pf_cpf_sv_dl4jets_test/'
    train_path = '/bigdata/shared/BumbleB/convert_20181121_ak8_80x_deepDoubleB_db_pf_cpf_sv_dl4jets_train_val/'
elif os.path.isdir('/eos/user/w/woodson/IN'):
    save_path = '/eos/user/w/woodson/IN/convert_20181121_ak8_80x_deepDoubleB_db_pf_cpf_sv_dl4jets_test/'
    train_path = '/eos/user/w/woodson/IN/convert_20181121_ak8_80x_deepDoubleB_db_pf_cpf_sv_dl4jets_train_val/'

spectators = ['fj_pt',
              'fj_eta',
              'fj_sdmass',
              'fj_n_sdsubjets',
              'fj_doubleb',
              'fj_tau21',
              'fj_tau32',
              'npv',
              'npfcands',
              'ntracks',
              'nsv']

truth = ['fj_truthHbb',
         'fj_truthQCD']

def main(args):
    test_spec_arrays = []
    train_spec_arrays = []
    test_truth_arrays = []
    train_truth_arrays = []
    
    for test_file in sorted(glob.glob(save_path + 'test_*_spectators_0.npy')):
        test_spec_arrays.append(np.load(test_file))
    test_spec = np.concatenate(test_spec_arrays)

    for test_file in sorted(glob.glob(save_path + 'test_*_truth_0.npy')):
        test_truth_arrays.append(np.load(test_file))
    test_truth = np.concatenate(test_truth_arrays)    

    for train_file in sorted(glob.glob(train_path + 'train_val_*_spectators_0.npy')):
        train_spec_arrays.append(np.load(train_file))
    train_spec = np.concatenate(train_spec_arrays)

    for train_file in sorted(glob.glob(train_path + 'train_val_*_truth_0.npy')):
        train_truth_arrays.append(np.load(train_file))
    train_truth = np.concatenate(train_truth_arrays)    

    del test_spec_arrays
    del train_spec_arrays
    del test_truth_arrays
    del train_truth_arrays

    test_spec = np.swapaxes(test_spec, 1, 2)
    train_spec = np.swapaxes(train_spec, 1, 2)
    #test_truth = np.swapaxes(test_truth, 1, 2)
    #train_truth = np.swapaxes(train_truth, 1, 2)
    print(test_spec.shape)
    print(train_spec.shape)
    print(test_truth.shape)
    print(train_truth.shape)

    fj_pt = test_spec[:,0,0]
    fj_eta = test_spec[:,1,0]
    fj_sdmass = test_spec[:,2,0]
    #no_undef = np.sum(test_truth,axis=1) == 1
    no_undef = fj_pt > -999 # no cut

    min_pt = -9999 #300
    max_pt = 999999 #2400
    min_eta = -9999 # no cut
    max_eta = 9999 # no cut
    min_msd = -9999 #40
    max_msd = 9999 #200
    
    test_spec = test_spec [ (fj_sdmass > min_msd) & (fj_sdmass < max_msd) & (fj_eta > min_eta) & (fj_eta < max_eta) & (fj_pt > min_pt) & (fj_pt < max_pt) & no_undef ]
    test_truth = test_truth [ (fj_sdmass > min_msd) & (fj_sdmass < max_msd) & (fj_eta > min_eta) & (fj_eta < max_eta) & (fj_pt > min_pt) & (fj_pt < max_pt) & no_undef  ]

    fj_pt = train_spec[:,0,0]
    fj_eta = train_spec[:,1,0]
    fj_sdmass = train_spec[:,2,0]
    no_undef = fj_pt > -999 # no cut

    train_spec = train_spec [ (fj_sdmass > min_msd) & (fj_sdmass < max_msd) & (fj_eta > min_eta) & (fj_eta < max_eta) & (fj_pt > min_pt) & (fj_pt < max_pt) & no_undef ]
    train_truth = train_truth [ (fj_sdmass > min_msd) & (fj_sdmass < max_msd) & (fj_eta > min_eta) & (fj_eta < max_eta) & (fj_pt > min_pt) & (fj_pt < max_pt) & no_undef  ]

    print("just signal")
    print(test_spec[test_truth[:,0]==1].shape)
    print(test_truth[test_truth[:,0]==1].shape)
    print(train_spec[train_truth[:,0]==1].shape)
    print(train_truth[train_truth[:,0]==1].shape)

    test_spec = test_spec.reshape(test_spec.shape[0],-1)
    train_spec = train_spec.reshape(train_spec.shape[0],-1)

    test_spec = test_spec.reshape(test_spec.shape[0],-1)
    train_spec = train_spec.reshape(train_spec.shape[0],-1)
    print(test_spec.shape)
    print(test_truth.shape)
    test_all = np.hstack([test_spec,test_truth])
    train_all = np.hstack([train_spec,train_truth])
    print(train_spec.shape)
    print(train_truth.shape)

    print(test_all.shape)
    print(train_all.shape)

    df_test = pd.DataFrame(test_all, columns = spectators+truth)
    df_test.to_pickle('%s/output_test.pkl'%args.outdir)

    df_train = pd.DataFrame(train_all, columns = spectators+truth)
    df_train.to_pickle('%s/output_train.pkl'%args.outdir)

if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()

    # Required positional arguments
    parser.add_argument("outdir", help="Required output directory")

    args = parser.parse_args()
    main(args)
