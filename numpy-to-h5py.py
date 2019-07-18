import os
import glob
import numpy as np
import h5py
import tqdm
import sys
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

batch_size = 100000
counter = 0
if os.path.isdir('/bigdata/shared/BumbleB'):
    save_path = '/bigdata/shared/BumbleB/convert_20181121_ak8_80x_deepDoubleB_db_pf_cpf_sv_dl4jets_test/'
elif os.path.isdir('/eos/user/w/woodson/IN'):
    save_path = '/eos/user/w/woodson/IN/convert_20181121_ak8_80x_deepDoubleB_db_pf_cpf_sv_dl4jets_test/'

train_0_arrays = []
train_1_arrays = []
train_2_arrays = []
train_3_arrays = []
truth_0_arrays = []
spec_0_arrays = []
weights_0_arrays = []

for i in range(len(glob.glob(save_path + 'train_val_*_features_0.npy'))):
    train_file = save_path + 'train_val_%i_features_0.npy'%i
    print("loading %s"%train_file)    
    train_0 = np.load(train_file, mmap_mode='r')
    train_file = save_path + 'train_val_%i_features_1.npy'%i
    print("loading %s"%train_file)    
    train_1 = np.load(train_file, mmap_mode='r')
    train_file = save_path + 'train_val_%i_features_2.npy'%i
    print("loading %s"%train_file)    
    train_2 = np.load(train_file, mmap_mode='r')
    train_file = save_path + 'train_val_%i_features_3.npy'%i
    print("loading %s"%train_file)    
    train_3 = np.load(train_file, mmap_mode='r')
    train_file = save_path + 'train_val_%i_truth_0.npy'%i
    print("loading %s"%train_file)    
    truth_0 = np.load(train_file, mmap_mode='r')
    train_file = save_path + 'train_val_%i_weights_0.npy'%i
    print("loading %s"%train_file)    
    weights_0 = np.load(train_file, mmap_mode='r')
    print("loading %s"%train_file)    
    train_file = save_path + 'train_val_%i_spectators_0.npy'%i
    spec_0 = np.load(train_file, mmap_mode='r')

    train_0 = np.swapaxes(train_0, 1, 2) 
    train_1 = np.swapaxes(train_1, 1, 2) 
    train_2 = np.swapaxes(train_2, 1, 2) 
    train_3 = np.swapaxes(train_3, 1, 2)

    for j in tqdm.tqdm(range(0, train_0.shape[0], batch_size)):
        h5 = h5py.File(save_path + "/newdata_" + str(counter) + ".h5", "w") #change this to change output location
        training_data = h5.create_group("training_subgroup")
        target_data = h5.create_group("target_subgroup")
        weight_data = h5.create_group("weight_subgroup")
        spec_data = h5.create_group("spectator_subgroup")
        training_data.create_dataset("training_0", data = train_0[j : j + batch_size])
        training_data.create_dataset("training_1", data = train_1[j : j + batch_size])
        training_data.create_dataset("training_2", data = train_2[j : j + batch_size])
        training_data.create_dataset("training_3", data = train_3[j : j + batch_size])
        target_data.create_dataset("target", data = truth_0[j : j + batch_size])
        weight_data.create_dataset("weights", data = weights_0[j : j + batch_size])
        spec_data.create_dataset("spectators", data = spec_0[j : j + batch_size])
        h5.close()
        counter += 1

