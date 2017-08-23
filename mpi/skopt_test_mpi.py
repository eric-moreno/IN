from __future__ import print_function

import os, itertools, random, glob, time, sys, ast, shutil
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
import argparse, json
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

parser.add_argument('--restart',
                    '-r',
                    dest = 'restart',
                    action = 'store_true'
                    )
parser.set_defaults(restart = False)

def save_state(path, optimizer, unfinished, finished, reported):
    dic = {'optimizer': optimizer,
           'unfinished': unfinished,
           'finished': finished,
           'reported': reported}
    pickle.dump(dic, open(path + '_tmp', 'wb'))
    os.rename(path + '_tmp', path)

def open_checkpoint(checkpoint):
    in_json  = open(checkpoint, 'r')
    in_dict = json.load(in_json)
    val_acc_vals = np.array(in_dict['val_acc_vals'])
    return val_acc_vals
    
def load_state(path):
    if os.path.exists(path):
        dic = pickle.load(open(path, 'rb'))
        optimizer, unfinished, finished, reported = dic['optimizer'], dic['unfinished'], \
                                                    dic['finished'], dic['reported']
    else:
        optimizer = Optimizer(
                        base_estimator=GaussianProcessRegressor(),
                        dimensions=[Integer(5, 20) for i in range(8)], acq_optimizer='sampling')
        unfinished = []
        finished = {}
        reported = {}
    return optimizer, unfinished, finished, reported

def get_path(data, checkpoint_path):
    return checkpoint_path + "-".join([str(i) for i in data]) + '/'

diff = lambda l1,l2: filter(lambda x: x not in l2, l1)

args = parser.parse_args()
n_epochs = args.epoch[0]
path = args.path[0]
restart = args.restart
checkpoint_path = path + 'checkpoints/'
if rank == 0:
    if restart:
        shutil.rmtree(checkpoint_path)
    kfolds = 10
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)
    optimizer, unfinished, finished, reported = load_state(checkpoint_path + 'optimizer.pkl')
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
                new_task = optimizer.ask()
                print("Starting: %s" % new_task)
                task_path = get_path(new_task, checkpoint_path)
                if not os.path.isdir(task_path):
                    os.makedirs(task_path)
                task_fold = [(new_task, i) for i in range(kfolds)]
                unfinished += task_fold
                task = unfinished[0]
            else: 
                task = l[0]
            working.append(task)
            save_state(checkpoint_path + 'optimizer.pkl', optimizer, unfinished, finished, reported)
            comm.send(task, dest=source, tag=START)
            #print("Sending task %s to worker %d at %s" % (task, source, time.asctime(time.localtime())))
        elif tag == DONE:
            #print("Got data %s from worker %d at %s" % (data, source, time.asctime(time.localtime())))
            graph_args, k = data
            if data in unfinished:
                unfinished.remove(data)
                working.remove(data)
                checkpoint = get_path(graph_args, checkpoint_path) + str(k) + '-best_checkpoint.json'
                val_acc_vals = open_checkpoint(checkpoint)
                inaccuracy = 100. - max(val_acc_vals)
                graph_args_key = str(graph_args)
                if graph_args_key in finished:
                    finished[graph_args_key][k] = inaccuracy
                else:
                    finished[graph_args_key] = {k: inaccuracy}
                if set(finished[graph_args_key].keys()) == set(range(kfolds)) and not graph_args_key in reported:
                    avg_inac = sum(finished[graph_args_key].values())/kfolds
                    print("Telling optimizer input: %s \n resulting in output: %s" % (graph_args, 100. - max(val_acc_vals)))
                    optimizer.tell(graph_args, 100. - max(val_acc_vals))
                    reported[graph_args_key] = True
            save_state(checkpoint_path + 'optimizer.pkl', optimizer, unfinished, finished, reported)
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
        args, k = task
        tag = status.Get_tag()
        if tag == START:
           # print ("Received parameters ",task,"to operate on")
            # Do the work here
            com = "python graph_train.py -e %s -a %s -p %s -k %s -t" % (n_epochs, 
                                                                     ' '.join([str(i) for i in args]), 
                                                                     path,
                                                                     k)
           # print ("Will execute the command: ", com)
            code = os.system(com)
            ## is there a way to catch that single.py exited without running a single epoch ? yes exit code 123
            comm.send(task, dest=0, tag=DONE)
        elif tag == EXIT:
            break

    comm.send(None, dest=0, tag=EXIT)
comm.Barrier() # wait for everybody to synchronize _here_
