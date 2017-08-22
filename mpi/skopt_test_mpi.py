from __future__ import print_function

import os, itertools, random, glob, time, sys, ast
import numpy as np

## MPI Setup
from mpi4py import MPI
comm = MPI.COMM_WORLD

size = comm.size        # total number of processes
rank = comm.Get_rank()# rank of this process
status = MPI.Status()   # get MPI status object

##skopt imports
from skopt import Optimizer
from skopt.learning import GaussianProcessRegressor
from skopt.space import Real, Integer
import matplotlib.pyplot as plt
## tags
READY=0
DONE=1
EXIT=2
START=3
NOTRAIN=4

import pickle
import argparse
parser = argparse.ArgumentParser(description='Get GNN parameters or saved state.')

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

def save_state(path, optimizer, unfinished):
    dic = {'optimizer': optimizer,
           'unfinished': unfinished}
    pickle.dump(dic, open(path + '_tmp', 'wb'))
    os.rename(path + '_tmp', path)
    
def load_state(path):
    if os.path.exists(path):
        dic = pickle.load(open(path, 'rb'))
        optimizer, unfinished = dic['optimizer'], dic['unfinished']
    else:
        optimizer = Optimizer(
                        base_estimator=GaussianProcessRegressor(),
                        dimensions=[Integer(5, 20) for i in range(8)], acq_optimizer='sampling')
        unfinished = []
    return optimizer, unfinished

def get_path(data, checkpoint_path):
    print(checkpoint_path)
    return checkpoint_path + "-".join([str(i) for i in data]) + '/'

diff = lambda l1,l2: filter(lambda x: x not in l2, l1)

args = parser.parse_args()
n_epochs = args.epoch[0]
path = args.path[0]
checkpoint_path = path + 'checkpoints/'
if rank == 0:
    optimizer, unfinished = load_state(checkpoint_path + 'optimizer.pkl')
    working = []
    num_workers = size - 1 # remove the master
    closed_workers = 0
    print("Master starting with %d workers" % num_workers)
    while closed_workers < num_workers:
        data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        source = status.Get_source()
        tag = status.Get_tag()
        if tag == READY:
            # Worker is ready, so send it a task
            ### keep on running until running out of time
            l = diff(unfinished, working)
            if l == []:
                task = optimizer.ask()
                unfinished.append(task)
            else: 
                task = l[0]
            working.append(task)
            save_state(checkpoint_path + 'optimizer.pkl', optimizer, unfinished)
            comm.send(task, dest=source, tag=START)
            print("Sending task %s to worker %d at %s"% (task, source, time.asctime(time.localtime())))
        elif tag == DONE:
            print("Got data %s from worker %d at %s" %(data, source, time.asctime(time.localtime())))
            unfinished.remove(data)
            working.remove(data)
            checkpoint = get_path(data, checkpoint_path) + 'best_checkpoint.txt'
            inf = open(checkpoint,'r')
            _, _, val_acc_vals, _ = [i.strip() for i in inf.readlines()]
            inf.close()
            val_acc_vals = np.array(ast.literal_eval(val_acc_vals))
            print("Telling optimizer input: %s \n resulting in output: %s" % (data, 100. - max(val_acc_vals)))
            optimizer.tell(data, 100. - max(val_acc_vals))
            save_state(checkpoint_path + 'optimizer.pkl', optimizer, unfinished)
        elif tag == EXIT:
            print("Worker %d exited at %s" % (source,time.asctime(time.localtime())))
            closed_workers += 1
else:
    # Worker processes execute code below
    name = MPI.Get_processor_name()
    print("I am a worker with rank %d on %s." % (rank, name))
    while True:
        comm.send(None, dest=0, tag=READY)
        task = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()
        if tag == START:
            print ("Received parameters ",task,"to operate on")
            # Do the work here
            com = "python graph_train.py -e %s -a %s -p %s" % (n_epochs, 
                                                               ' '.join([str(i) for i in task]), 
                                                               path)
            print ("Will execute the command: ", com)
            code = os.system(com)
            ## is there a way to catch that single.py exited without running a single epoch ? yes exit code 123
            comm.send(task, dest=0, tag=DONE)
        elif tag == EXIT:
            break

    comm.send(None, dest=0, tag=EXIT)
comm.Barrier() # wait for everybody to synchronize _here_
