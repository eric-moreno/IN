from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd.variable import *
import torch.optim as optim
import os
import numpy as np
import pandas as pd
import util
import itertools
class GraphNet(nn.Module):
    def __init__(self, 
                 n_constituents, 
                 n_targets, 
                 params,
                 De = 5, Do = 6,
                 hiddenr1 = 10, hiddeno1 = 10, hiddenc1 = 10,
                 hiddenr2 = 5, hiddeno2 = 5, hiddenc2 = 5):
        super(GraphNet, self).__init__()
        self.hiddenr1 = hiddenr1
        self.hiddenr2 = hiddenr2
        self.hiddeno1 = hiddeno1
        self.hiddeno2 = hiddeno2
        self.hiddenc1 = hiddenc1
        self.hiddenc2 = hiddenc2
        self.P = len(params)
        self.N = n_constituents
        self.Nr = self.N * (self.N - 1)
        self.Dr = 0
        self.De = De
        self.Dx = 0
        self.Do = Do
        self.n_targets = n_targets
        self.assign_matrices()
        self.Ra = Variable(torch.ones(self.Dr, self.Nr))
        self.fr1 = nn.Linear(2 * self.P + self.Dr, self.hiddenr1).cuda()
        self.fr2 = nn.Linear(self.hiddenr1, self.hiddenr2).cuda()
        self.fr3 = nn.Linear(self.hiddenr2, self.De).cuda()
        self.fo1 = nn.Linear(self.P + self.Dx + self.De, self.hiddeno1).cuda()
        self.fo2 = nn.Linear(self.hiddeno1, self.hiddeno2).cuda()
        self.fo3 = nn.Linear(self.hiddeno2, self.Do).cuda()
        self.fc1 = nn.Linear(self.Do * self.N, self.hiddenc1).cuda()
        self.fc2 = nn.Linear(self.hiddenc1, self.hiddenc2).cuda()
        self.fc3 = nn.Linear(self.hiddenc2, self.n_targets).cuda()
    
    def assign_matrices(self):
        self.Rr = torch.zeros(self.N, self.Nr)
        self.Rs = torch.zeros(self.N, self.Nr)
        receiver_sender_list = [i for i in itertools.product(range(self.N), range(self.N)) if i[0]!=i[1]]
        for i, (r, s) in enumerate(receiver_sender_list):
            self.Rr[r, i] = 1
            self.Rs[s, i] = 1
        self.Rr = Variable(self.Rr).cuda()
        self.Rs = Variable(self.Rs).cuda()
        
    def forward(self, x):
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
        C = torch.cat([x, Ebar], 1)
        C = torch.transpose(C, 1, 2).contiguous()
        ### Second MLP ###
        C = nn.functional.relu(self.fo1(C.view(-1, self.P + self.Dx + self.De)))
        C = nn.functional.relu(self.fo2(C))
        O = nn.functional.relu(self.fo3(C).view(-1, self.N, self.Do))
        del C
        ### Classification MLP ###
        N = nn.functional.relu(self.fc1(O.view(-1, self.Do * self.N)))
        del O
        N = nn.functional.relu(self.fc2(N))
        N = nn.functional.relu(self.fc3(N))
        return N

    def tmul(self, x, y):  #Takes (I * J * K)(K * L) -> I * J * L 
        x_shape = x.size()
        y_shape = y.size()
        return torch.mm(x.view(-1, x_shape[2]), y).view(-1, x_shape[1], y_shape[1])