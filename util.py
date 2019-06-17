import numpy as np
import os, sys, ast, type_func, pickle
import pandas as pd
import h5py
from random import random
import itertools

params = ['Px', 'Py', 'Pz', 'PT', 'E', 'D0', 'DZ', 'X', 'Y', 'Z']
max_len = 100
nothing = "[]"
jet_type_num = {0:'higgs', 1: 'top', 2: 'Z', 3: 'W+', 4: 'strange', 100: nothing}
jet_type = {v: k for k, v in jet_type_num.items()}
POI = {6:'top', 25:'higgs', 23:'Z', 3: 'strange', 24:'W+'}

def print_attrs(object):
    print("\n".join(dir(object)))

def print_trefarray(array):
    num_entries = array.GetEntries()
    print("[")
    for i in range(num_entries):
        print(" ", str(array.At(i)) + ",\n",)
    print(" ", array.At(num_entries - 1))
    print("]")

def tref_array_to_numpy(array):
    num_entries = array.GetEntries()
    return np.array([array.At(i) for i in range(num_entries)])

def get_MOI(particle, branch_particle, count = 0, POI = POI):
    if particle.M1 == 0 or branch_particle.At(particle.M1) == particle:
        return nothing, count
    if particle.PID in POI:
        return POI[particle.PID], count
    else:
        try:
            m1, c1 = get_MOI(branch_particle.At(particle.M1), branch_particle, count = count + 1)
        except:
            m1, c1 = nothing, 0
        try:
            m2, c2 = get_MOI(branch_particle.At(particle.M2), branch_particle, count = count + 1)
        except:
            m2, c2 = nothing, 0
        if m1 == nothing:
            return m2, c2
        return m1, c1

def constituent_method(constituent, event, jet, moms):
    s = str(constituent)
    if "Muon" in s:
        d = muon_to_dict(constituent, event, jet)
    elif "Track" in s:
        d = track_to_dict(constituent, event, jet)
    elif "GenParticle" in s:
        d = particle_to_dict(constituent, event, jet)
    else:
        return empty_dict(constituent, event, jet)
    d['parents'] = moms[jet]
    return d

def empty_dict(particle, event, jet):
    return {}

def particle_to_dict(particle, event, jet,  
                     params = ["Px", "Py", "Pz", 
                               "PID", "E", "P", "T", "M1", "M2", "D1", "D2", 
                               "D0", "DZ", "X", "Y", "Z",
                               "PT"]):
    d = dict((param, getattr(particle, param)) for param in params)
    d['event'] = event
    d['jet'] = jet
    d['track'] = False
    d['muon'] = False
    return d

def track_to_dict(track, event, jet, params = ["P", "PID"]):
    p = track.Particle.GetObject()
    if "GenParticle" in str(p):
        d = particle_to_dict(p, event, jet)
    else:
        d = dict((param, getattr(track, param)) for param in params)
        d['mom'] = nothing
    d['event'] = event
    d['jet'] = jet
    d['muon'] = False
    d['track'] = True
    d['track_id'] = getattr(track, "PID")
    return d

def muon_to_dict(muon, event, jet, params = ["PT"]):
    p = muon.Particle.GetObject()
    if not "0x0" in str(p):
        d = particle_to_dict(p, event, jet)
    else:
        d = dict((param, getattr(track, param)) for param in params)
        d['mom'] = nothing
    d['event'] = event
    d['jet'] = jet
    d['track'] = False
    d['muon'] = True
    #print(muon.Particle.GetObject())
    return d

def get_mother_list(particle, branch_particle, mom_list = []):
    m2 = particle.M2
    m1 = particle.M1
    if m1 >= branch_particle.GetEntries() or m2 >= branch_particle.GetEntries():
        return sorted(list(set(mom_list)))
    if particle == branch_particle.At(particle.M1):
        return sorted(list(set(mom_list)))
    if m1 == 0 and m2 == 0:
        return sorted(list(set(mom_list)))
    if m1 in mom_list and m2 in mom_list:
        return sorted(list(set(mom_list)))
    mom_list.insert(0, m1)
    mom_list.insert(0, m2)
    mom_list = list(set(mom_list)) 
    new_list = get_mother_list(branch_particle.At(m1), branch_particle, mom_list) + \
               get_mother_list(branch_particle.At(m2), branch_particle, mom_list)
    return sorted(list(set(new_list)))

def not_tobject(index, branch_particle):
    if "TObject" in str(type(branch_particle.At(index))):
        return False
    return True

def get_jet_parents(jet, branch_particle):
    """Get a fine-grained parent"""
    #get list of parents
    constituents = tref_array_to_numpy(jet.Constituents)
    moms = [get_mother_list(particle, branch_particle) for particle in constituents]
    #flatten list
    moms = sorted(list(set(itertools.chain.from_iterable(moms))))
    num_entries = branch_particle.GetEntries()
    moms = [i for i in moms if i < num_entries]
    pids = [branch_particle.At(i).PID for i in moms if not_tobject(i, branch_particle)]
    jet_parents = []
    a= jet_parents.append
    if 25 in pids:
        a('H')
    if 6 in pids:
        a('t')
    if pids.count(5) >= 2:
        a('b')
        a('b')
    if pids.count(5) == 1:
        a('b')
    if 24 in pids:
        a('W+')
    if pids.count(4) >= 2:
        a('c')
        a('c')
    if pids.count(4) == 1:
        a('c')
    if pids.count(3) >= 2:
        a('s')
        a('s')
    if pids.count(3) == 1:
        a('s')
    if 23 in pids:
        a('Z')
    jet_parents = sorted(jet_parents)
    return str(jet_parents)

def assign_parent_list_to_dict(dic, parent):
    dic['parents'] = parent
    return dic

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '#'):
    """ Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    sys.stdout.write('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix))
    sys.stdout.flush()
    # Print New Line on Complete
    if iteration == total: 
        print()

def df_filter_nothing(df):
    dfs = filter_jet_list(df_get_jet_list(df))
    return pd.concat(dfs)

def combine_dfs(dfs):
    count = 0
    for df in dfs:
        df['njet'] += count
        count += sorted(np.unique(df['njet'].values))[-1]
    df = pd.concat(dfs)
    return df

def h5_to_target(fname, output = None, params = params, max_len = max_len):
    df = h5_to_df(fname)
    return df_to_target(df, output, params, max_len = max_len)

def h5_to_df(fname, jet_dict_file = None):
    print("reading file")
    df = pd.read_hdf(fname)
    print("Generate dictionary")
    types = sorted(list(set(df.parents.values)))
    if jet_dict_file == None:
        jet_dict = {}
        for i in types:
            #l = int(raw_input(i + ": "))
            j = ast.literal_eval(i)
            l = type_func.get_type(j)
            print(j, l)
            jet_dict[i] = l
        pickle.dump(jet_dict, open('jet_dict.pkl', 'wb'))
    else:
        jet_dict = pickle.load(open(jet_dict_file, 'rb'))
    print("Assigning jet_type...")
    df['mom'] = df['parents'].map(jet_dict)
    df['count'] = df.groupby('njet')['parents'].transform('count')
    return df

def pad_values(vals, val = 0, max_len = max_len):
    sr, sc = vals.shape
    if max_len - sr > 0:
        return np.pad(vals, ((0, max_len - sr), (0, 0)), mode='constant')
    return vals[:max_len, :]

def print_accuracy( p, target ):
    p_cat = np.argmax(p,axis=1)
    test_target = np.argmax(target, axis = 1)
    print("Fraction of good prediction")
    print(len(np.where( p_cat == test_target)[0]))
    print(len(np.where( p_cat == test_target )[0])/float(len(p_cat)),"%")

def accuracy(p, target):
    p_cat = np.argmax(p,axis=1)
    test_target = np.argmax(target, axis = 1)
    return len(np.where( p_cat == test_target)[0])/float(len(p_cat))

def df_njet_index(df):
    """Combine event # and jet #"""
    events_jets = df[['event', 'jet']].values
    njet = np.zeros(events_jets.shape[0])
    prev = np.array([-1, -1])
    count = -1
    for ind, val in enumerate(events_jets):
        if any(prev != val):
            count += 1
            prev = val
        njet[ind] = count
    df['njet'] = njet
    return df

def df_get_jet_list(df):
    """Return a list of dataframes based on jet number"""
    groups = df.groupby('njet')
    print("Generated groupby object")
    dfs = [groups.get_group(i) for i in groups.groups.keys()] 
    return dfs
    
def df_to_target(df, output = None, params = params, max_len = max_len):
    df = df.sort_values(['njet', 'D0', 'DZ', 'PT'],ascending = False)
    numpy_vals = df[['njet'] + params].values
    moms = df[['njet', 'mom']].values 
    ma = max(moms[:, 1])
    training = np.array([pad_values(i[:, 1:], max_len = max_len) 
                            for i in np.split(numpy_vals, np.where(np.diff(numpy_vals[:,0]))[0]+1)])
    training_target = np.array([get_list_from_num(i[0, 1], length = 1 + ma)
                        for i in np.split(moms, np.where(np.diff(moms[:, 0]))[0] + 1)])
    return training, training_target
    print("lookup")
    dfs = df.groupby('njet')
    jets = df.njet.unique()
    jet_sub = np.random.choice(jets, 100, replace = False)
    print("got jets")
    training = np.array([pad_values(dfs.get_group(i)[params].sort_values(['D0', 'DZ', 'PT'], 
                                                              ascending = False).values) 
                                                                         for i in jet_sub])
    print("got training")
    if output == None:
        training_target = np.array([get_list_from_num(get_jet_num(dfs.get_group(i))) 
                                        for i in jet_sub])
    else:
        training_target = np.array([output for i in range(len(training))])
    print("to numpy")
    #training = np.array([pad_values(i) for i in training])
    return training, training_target
               
def filter_jet_list(dfs):
    """Remove all the Nothings"""
    dfs = [i for i in dfs if i.parents.values[0] !=nothing]
    return dfs

def get_list_from_num(num, length = (len(jet_type_num) - 1)):
    l = np.zeros(length)
    l[int(num)] = 1
    return l

def get_jet_num(df):
    return df.mom.values[0] 
               
def assign_jet_type(df, jet_dict):
    """Assign most appropriate jet type"""
    jet_type_choice = jet_dict[df.parents.values[0]]
    df['mom'] = jet_type_choice
    return df

def make_test_split(training, target, test_size = 200):
    """Split training/target into training/target and test/target"""
    print(training.shape, target.shape)
    num = training.shape[0]
    indices = np.random.choice(range(num), test_size, replace = False)
    test = training[indices]
    test_target = target[indices]
    training = np.delete(training, indices, axis = 0)
    target = np.delete(target, indices, axis = 0)
    return training, target, test, test_target

def get_training_target_sample(training, target, sample_size):
    indices = np.random.choice(range(training.shape[0]), sample_size, replace = False)
    ntraining = training[indices]
    ntarget = target[indices]
    return ntraining, ntarget

def shuffle_together(training, target):
    p = np.random.permutation(len(training))
    return training[p], target[p]

def combine_sets(sets, sample_size = 10000):
    """Combine a list of (training, target, test, test_target) sets"""
    a = min([i[0].shape[0] for i in sets])
    sample_size = min(a, sample_size)
    samples = [get_training_target_sample(i[0], i[1], sample_size) for i in sets]
    ntraining = np.concatenate([i[0] for i in samples])
    ntarget = np.concatenate([i[1] for i in samples])
    ntest = np.concatenate([i[2] for i in sets])
    ntest_target = np.concatenate([i[3] for i in sets])
    ntraining, ntarget = shuffle_together(ntraining, ntarget)
    ntest, ntest_target = shuffle_together(ntest, ntest_target)
    return ntraining, ntarget, ntest, ntest_target

def generate_training_set(files):
    targets = [h5_to_target(i) for i in files]
    splits = [make_test_split(*i) for i in targets]
    training, training_target, test, test_target = combine_sets(splits, sample_size = 5000)
    return training, training_target, test, test_target
