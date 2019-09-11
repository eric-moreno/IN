import matplotlib as mpl
mpl.use('agg')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
import torch
import glob
import tqdm
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import rc
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

def main(args):
    test_spec_arrays = []
    train_spec_arrays = []
    
    for test_file in sorted(glob.glob(save_path + 'test_*_spectators_0.npy')):
        test_spec_arrays.append(np.load(test_file))
    test_spec = np.concatenate(test_spec_arrays)

    for train_file in sorted(glob.glob(train_path + 'train_val_*_spectators_0.npy')):
        train_spec_arrays.append(np.load(train_file))
    train_spec = np.concatenate(train_spec_arrays)

    del test_spec_arrays
    del train_spec_arrays

    test_spec = np.swapaxes(test_spec, 1, 2)
    train_spec = np.swapaxes(train_spec, 1, 2)
    print(test_spec.shape)
    print(train_spec.shape)

    df_test = pd.DataFrame(test_spec.reshape(test_spec.shape[0],-1), columns = spectators)
    df_test.to_pickle('%s/output_test.pkl'%args.outdir)

    df_train = pd.DataFrame(train_spec.reshape(train_spec.shape[0],-1), columns = spectators)
    df_train.to_pickle('%s/output_train.pkl'%args.outdir)

if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()

    # Required positional arguments
    parser.add_argument("outdir", help="Required output directory")

    args = parser.parse_args()
    main(args)
