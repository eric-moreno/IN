from __future__ import print_function
import os
import numpy as np
import pandas as pd
import util
import setGPU
import glob
import sys
import tqdm
import argparse
import keras
from keras.layers import Input
from keras.callbacks import ModelCheckpoint, EarlyStopping

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
test_path = '/storage/group/gpu/bigdata/BumbleB/convert_20181121_ak8_80x_deepDoubleB_db_pf_cpf_sv_dl4jets_test/'
train_path = '/storage/group/gpu/bigdata/BumbleB/convert_20181121_ak8_80x_deepDoubleB_db_pf_cpf_sv_dl4jets_train_val/'
NBINS = 40 # number of bins for loss function
MMAX = 200. # max value
MMIN = 40. # min value

N = 60 # number of charged particles
N_sv = 5 # number of SVs 
n_targets = 2 # number of classes

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

def main(args):
    """ Main entry point of the app """
    
    #Convert two sets into two branch with one set in both and one set in only one (Use for this file)
    params = params_2
    params_sv = params_3
    
    from data import H5Data
    files = glob.glob(train_path + "/newdata_*.h5")
    files_val = files[:5] # take first 5 for validation
    files_train = files[5:] # take rest for training
    
    label = 'new'
    outdir = args.outdir
    os.system('mkdir -p %s'%outdir)

    batch_size = 1024
    data_train = H5Data(batch_size = batch_size,
                        cache = None,
                        preloading=0,
                        features_name='training_subgroup', 
                        labels_name='target_subgroup',
                        spectators_name='spectator_subgroup')
    data_train.set_file_names(files_train)
    data_val = H5Data(batch_size = batch_size,
                      cache = None,
                      preloading=0,
                      features_name='training_subgroup', 
                      labels_name='target_subgroup',
                      spectators_name='spectator_subgroup')
    data_val.set_file_names(files_val)

    n_val=data_val.count_data()
    n_train=data_train.count_data()
    print("val data:", n_val)
    print("train data:", n_train)

    from ddb import model_DeepDoubleXReference
    keras_model = model_DeepDoubleXReference(inputs = [Input(shape=(N,len(params))),Input(shape=(N_sv,len(params_sv)))],
                                             num_classes = n_targets, scale_hidden = 2, 
                                             hlf_input = None, datasets = ['cpf', 'sv'])
    
    keras_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

    keras_model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=20)
    model_checkpoint = ModelCheckpoint('%s/keras_model_best.h5'%outdir, monitor='val_loss', save_best_only=True)
    callbacks = [early_stopping, model_checkpoint]
    keras_model.fit_generator(data_train.inf_generate_data_keras(),
                              validation_data = data_val.inf_generate_data_keras(),
                              epochs=200, 
                              steps_per_epoch=np.ceil(n_train/batch_size), 
                              validation_steps=np.ceil(n_val/batch_size), 
                              callbacks = callbacks)
    

if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()
    
    # Required positional arguments
    parser.add_argument("outdir", help="Required output directory")
    
    args = parser.parse_args()
    main(args)
