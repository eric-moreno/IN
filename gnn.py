import itertools
import torch
import torch.nn as nn
from torch.autograd.variable import *
import torch.optim as optim

class GraphNet(nn.Module):
    def __init__(self, n_constituents, n_targets, params, hidden, n_vertices, params_v, vv_branch=False, De=5, Do=6, softmax=False):
        super(GraphNet, self).__init__()
        self.hidden = int(hidden)
        self.P = params
        self.N = n_constituents
        self.S = params_v
        self.Nv = n_vertices
        self.Nr = self.N * (self.N - 1)
        self.Nt = self.N * self.Nv
        self.Ns = self.Nv * (self.Nv - 1)
        self.Dr = 0
        self.De = De
        self.Dx = 0
        self.Do = Do
        self.n_targets = n_targets
        self.assign_matrices()
        self.assign_matrices_SV()
        self.vv_branch = vv_branch
        self.softmax = softmax
        if self.vv_branch:
            self.assign_matrices_SVSV()
        
        self.Ra = torch.ones(self.Dr, self.Nr)
        self.fr1 = nn.Linear(2 * self.P + self.Dr, self.hidden).cuda()
        self.fr2 = nn.Linear(self.hidden, int(self.hidden/2)).cuda()
        self.fr3 = nn.Linear(int(self.hidden/2), self.De).cuda()
        self.fr1_pv = nn.Linear(self.S + self.P + self.Dr, self.hidden).cuda()
        self.fr2_pv = nn.Linear(self.hidden, int(self.hidden/2)).cuda()
        self.fr3_pv = nn.Linear(int(self.hidden/2), self.De).cuda()
        if self.vv_branch:
            self.fr1_vv = nn.Linear(2 * self.S + self.Dr, self.hidden).cuda()
            self.fr2_vv = nn.Linear(self.hidden, int(self.hidden/2)).cuda()
            self.fr3_vv = nn.Linear(int(self.hidden/2), self.De).cuda()
        self.fo1 = nn.Linear(self.P + self.Dx + (2 * self.De), self.hidden).cuda()
        self.fo2 = nn.Linear(self.hidden, int(self.hidden/2)).cuda()
        self.fo3 = nn.Linear(int(self.hidden/2), self.Do).cuda()
        if self.vv_branch:
            self.fo1_v = nn.Linear(self.S + self.Dx + (2 * self.De), self.hidden).cuda()
            self.fo2_v = nn.Linear(self.hidden, int(self.hidden/2)).cuda()
            self.fo3_v = nn.Linear(int(self.hidden/2), self.Do).cuda()

        if self.vv_branch:
            #self.fc_1 = nn.Linear(2*self.Do, self.Do).cuda()
            #self.fc_2 = nn.Linear(self.Do, int(self.Do/2)).cuda()
            #self.fc_3 = nn.Linear(int(self.Do/2), self.n_targets).cuda()
            self.fc_fixed = nn.Linear(2*self.Do, self.n_targets).cuda()
        else:
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

    def assign_matrices_SVSV(self):
        self.Rl = torch.zeros(self.Nv, self.Ns)
        self.Ru = torch.zeros(self.Nv, self.Ns)
        receiver_sender_list = [i for i in itertools.product(range(self.Nv), range(self.Nv)) if i[0]!=i[1]]
        for i, (l, u) in enumerate(receiver_sender_list):
            self.Rl[l, i] = 1
            self.Ru[u, i] = 1
        self.Rl = (self.Rl).cuda()
        self.Ru = (self.Ru).cuda()

    def forward(self, x, y):
        ###PF Candidate - PF Candidate###
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
        Ebar_pp = self.tmul(E, torch.transpose(self.Rr, 0, 1).contiguous())
        del E
        
        ####Secondary Vertex - PF Candidate### 
        Ork = self.tmul(x, self.Rk)
        Orv = self.tmul(y, self.Rv)
        B = torch.cat([Ork, Orv], 1)
        ### First MLP ###
        B = torch.transpose(B, 1, 2).contiguous()
        B = nn.functional.relu(self.fr1_pv(B.view(-1, self.S + self.P + self.Dr)))
        B = nn.functional.relu(self.fr2_pv(B))
        E = nn.functional.relu(self.fr3_pv(B).view(-1, self.Nt, self.De))
        del B
        E = torch.transpose(E, 1, 2).contiguous()
        Ebar_pv = self.tmul(E, torch.transpose(self.Rk, 0, 1).contiguous())
        Ebar_vp = self.tmul(E, torch.transpose(self.Rv, 0, 1).contiguous())
        del E

        ###Secondary vertex - secondary vertex###
        if self.vv_branch:
            Orl = self.tmul(y, self.Rl)
            Oru = self.tmul(y, self.Ru)
            B = torch.cat([Orl, Oru], 1)
            ### First MLP ###
            B = torch.transpose(B, 1, 2).contiguous()
            B = nn.functional.relu(self.fr1_vv(B.view(-1, 2 * self.S + self.Dr)))
            B = nn.functional.relu(self.fr2_vv(B))
            E = nn.functional.relu(self.fr3_vv(B).view(-1, self.Ns, self.De))
            del B
            E = torch.transpose(E, 1, 2).contiguous()
            Ebar_vv = self.tmul(E, torch.transpose(self.Rl, 0, 1).contiguous())
            del E

        ####Final output matrix for particles###
        C = torch.cat([x, Ebar_pp, Ebar_pv], 1)
        del Ebar_pp
        del Ebar_pv
        C = torch.transpose(C, 1, 2).contiguous()
        ### Second MLP ###
        C = nn.functional.relu(self.fo1(C.view(-1, self.P + self.Dx + (2 * self.De))))
        C = nn.functional.relu(self.fo2(C))
        O = nn.functional.relu(self.fo3(C).view(-1, self.N, self.Do))
        del C

        if self.vv_branch:
            ####Final output matrix for particles### 
            C = torch.cat([y, Ebar_vv, Ebar_vp], 1)
            del Ebar_vv
            del Ebar_vp
            C = torch.transpose(C, 1, 2).contiguous()
            ### Second MLP ###
            C = nn.functional.relu(self.fo1_v(C.view(-1, self.S + self.Dx + (2 * self.De))))
            C = nn.functional.relu(self.fo2_v(C))
            O_v = nn.functional.relu(self.fo3_v(C).view(-1, self.Nv, self.Do))
            del C
        
        #Taking the sum of over each particle/vertex
        N = torch.sum(O, dim=1)
        del O
        if self.vv_branch:
            N_v = torch.sum(O_v,dim=1)
            del O_v
        
        ### Classification MLP ###
        if self.vv_branch:
            #N = nn.functional.relu(self.fc_1(torch.cat([N, N_v],1)))
            #N = nn.functional.relu(self.fc_2(N))
            #N = self.fc_3(N)
            N =self.fc_fixed(torch.cat([N, N_v],1))
        else:
            N = self.fc_fixed(N)

        if self.softmax:
            N = nn.Softmax(dim=-1)(N)

        return N 
            
    def tmul(self, x, y):  #Takes (I * J * K)(K * L) -> I * J * L 
        x_shape = x.size()
        y_shape = y.size()
        return torch.mm(x.view(-1, x_shape[2]), y).view(-1, x_shape[1], y_shape[1])

class GraphNetAdv(GraphNet):
    def __init__(self, n_constituents, n_targets, params, hidden, n_vertices, params_v, vv_branch=False, De=5, Do=6):
        super(GraphNetAdv, self).__init__(n_constituents, n_targets, params, hidden, n_vertices, params_v, vv_branch, De, Do)

    def forward(self, x, y):
        ###PF Candidate - PF Candidate###
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
        Ebar_pp = self.tmul(E, torch.transpose(self.Rr, 0, 1).contiguous())
        del E
        
        ####Secondary Vertex - PF Candidate### 
        Ork = self.tmul(x, self.Rk)
        Orv = self.tmul(y, self.Rv)
        B = torch.cat([Ork, Orv], 1)
        ### First MLP ###
        B = torch.transpose(B, 1, 2).contiguous()
        B = nn.functional.relu(self.fr1_pv(B.view(-1, self.S + self.P + self.Dr)))
        B = nn.functional.relu(self.fr2_pv(B))
        E = nn.functional.relu(self.fr3_pv(B).view(-1, self.Nt, self.De))
        del B
        E = torch.transpose(E, 1, 2).contiguous()
        Ebar_pv = self.tmul(E, torch.transpose(self.Rk, 0, 1).contiguous())
        Ebar_vp = self.tmul(E, torch.transpose(self.Rv, 0, 1).contiguous())
        del E

        ###Secondary vertex - secondary vertex###
        if self.vv_branch:
            Orl = self.tmul(y, self.Rl)
            Oru = self.tmul(y, self.Ru)
            B = torch.cat([Orl, Oru], 1)
            ### First MLP ###
            B = torch.transpose(B, 1, 2).contiguous()
            B = nn.functional.relu(self.fr1_vv(B.view(-1, 2 * self.S + self.Dr)))
            B = nn.functional.relu(self.fr2_vv(B))
            E = nn.functional.relu(self.fr3_vv(B).view(-1, self.Ns, self.De))
            del B
            E = torch.transpose(E, 1, 2).contiguous()
            Ebar_vv = self.tmul(E, torch.transpose(self.Rl, 0, 1).contiguous())
            del E

        ####Final output matrix for particles###
        C = torch.cat([x, Ebar_pp, Ebar_pv], 1)
        del Ebar_pp
        del Ebar_pv
        C = torch.transpose(C, 1, 2).contiguous()
        ### Second MLP ###
        C = nn.functional.relu(self.fo1(C.view(-1, self.P + self.Dx + (2 * self.De))))
        C = nn.functional.relu(self.fo2(C))
        O = nn.functional.relu(self.fo3(C).view(-1, self.N, self.Do))
        del C

        if self.vv_branch:
            ####Final output matrix for particles### 
            C = torch.cat([y, Ebar_vv, Ebar_vp], 1)
            del Ebar_vv
            del Ebar_vp
            C = torch.transpose(C, 1, 2).contiguous()
            ### Second MLP ###
            C = nn.functional.relu(self.fo1_v(C.view(-1, self.S + self.Dx + (2 * self.De))))
            C = nn.functional.relu(self.fo2_v(C))
            O_v = nn.functional.relu(self.fo3_v(C).view(-1, self.Nv, self.Do))
            del C
        
        #Taking the sum of over each particle/vertex
        N = torch.sum(O, dim=1)
        del O
        if self.vv_branch:
            N_v = torch.sum(O_v,dim=1)
            N = torch.cat([N, N_v])
            del O_v
        
        ### Classification MLP ###
        F = self.fc_fixed(N)

        return (F, N)

# Architecture that excludes Secondary Vertices branch from Interaction network
class GraphNetnoSV(nn.Module):
    def __init__(self, n_constituents, n_targets, params, hidden, De=5, Do=6, softmax=False):
        super(GraphNetnoSV, self).__init__()
        self.hidden = int(hidden)
        self.P = params
        self.N = n_constituents
        self.Nr = self.N * (self.N - 1)
        self.Nt = self.N * self.Nv
        self.Ns = self.Nv * (self.Nv - 1)
        self.Dr = 0
        self.De = De
        self.Dx = 0
        self.Do = Do
        self.n_targets = n_targets
        self.assign_matrices()
           
        self.Ra = torch.ones(self.Dr, self.Nr)
        self.fr1 = nn.Linear(2 * self.P + self.Dr, self.hidden).cuda()
        self.fr2 = nn.Linear(self.hidden, int(self.hidden/2)).cuda()
        self.fr3 = nn.Linear(int(self.hidden/2), self.De).cuda()
        self.fr1_pv = nn.Linear(self.S + self.P + self.Dr, self.hidden).cuda()
        self.fr2_pv = nn.Linear(self.hidden, int(self.hidden/2)).cuda()
        self.fr3_pv = nn.Linear(int(self.hidden/2), self.De).cuda()
        
        self.fo1 = nn.Linear(self.P + self.Dx + (self.De), self.hidden).cuda()
        self.fo2 = nn.Linear(self.hidden, int(self.hidden/2)).cuda()
        self.fo3 = nn.Linear(int(self.hidden/2), self.Do).cuda()
        
        self.fc_fixed = nn.Linear(self.Do, self.n_targets).cuda()
            
    def assign_matrices(self):
        self.Rr = torch.zeros(self.N, self.Nr)
        self.Rs = torch.zeros(self.N, self.Nr)
        receiver_sender_list = [i for i in itertools.product(range(self.N), range(self.N)) if i[0]!=i[1]]
        for i, (r, s) in enumerate(receiver_sender_list):
            self.Rr[r, i] = 1
            self.Rs[s, i] = 1
        self.Rr = (self.Rr).cuda()
        self.Rs = (self.Rs).cuda()

    def forward(self, x):
        ###PF Candidate - PF Candidate###
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
        Ebar_pp = self.tmul(E, torch.transpose(self.Rr, 0, 1).contiguous())
        del E
        

        ####Final output matrix for particles###
        

        C = torch.cat([x, Ebar_pp], 1)
        del Ebar_pp
        C = torch.transpose(C, 1, 2).contiguous()
        ### Second MLP ###
        C = nn.functional.relu(self.fo1(C.view(-1, self.P + self.Dx + (self.De))))
        C = nn.functional.relu(self.fo2(C))
        O = nn.functional.relu(self.fo3(C).view(-1, self.N, self.Do))
        del C

        
        #Taking the sum of over each particle/vertex
        N = torch.sum(O, dim=1)
        del O
        
        ### Classification MLP ###

        N = self.fc_fixed(N)

        if softmax:
            N = nn.Softmax(dim=-1)(N)

        return N 
            
    def tmul(self, x, y):  #Takes (I * J * K)(K * L) -> I * J * L 
        x_shape = x.size()
        y_shape = y.size()
        return torch.mm(x.view(-1, x_shape[2]), y).view(-1, x_shape[1], y_shape[1])


#Architecture that includes neutral pfcandidates - not enabled by default

class GraphNetNeutral(nn.Module):
    def __init__(self, n_constituents_charged, n_constituents_neutral, n_targets, params_charged, params_neutral, hidden, n_vertices, params_v, sv_branch=True, vv_branch=False, nn_branch=False, De=5, Do=6):
        super(GraphNetNeutral, self).__init__()
        self.hidden = int(hidden)
        self.P = params_charged
        self.N = n_constituents_charged
        self.S = params_v
        self.T = params_neutral
        self.Nv = n_vertices
        self.Nneu = n_constituents_neutral
        self.Ne = self.Nneu * (self.Nneu - 1)
        self.Nr = self.N * (self.N - 1)
        self.Nn = self.N * self.Nneu
        self.Nt = self.N * self.Nv
        self.Ns = self.Nv * (self.Nv - 1)
        
        self.Dr = 0
        self.De = De
        self.Dx = 0
        self.Do = Do
        self.n_targets = n_targets
        self.assign_matrices()
        self.assign_matrices_SV()
        self.assign_matrices_neutral()
        self.vv_branch = vv_branch
        self.nn_branch = nn_branch
        if self.vv_branch:
            self.assign_matrices_SVSV()
        if self.nn_branch: 
            self.assign_matrices_neutralneutral()
        
        self.Ra = torch.ones(self.Dr, self.Nr)
        self.fr1 = nn.Linear(2 * self.P + self.Dr, self.hidden).cuda()
        self.fr2 = nn.Linear(self.hidden, int(self.hidden/2)).cuda()
        self.fr3 = nn.Linear(int(self.hidden/2), self.De).cuda()
        self.fr1_pv = nn.Linear(self.S + self.P + self.Dr, self.hidden).cuda()
        self.fr2_pv = nn.Linear(self.hidden, int(self.hidden/2)).cuda()
        self.fr3_pv = nn.Linear(int(self.hidden/2), self.De).cuda()
        self.fr1_pn = nn.Linear(self.T + self.P + self.Dr, self.hidden).cuda()
        self.fr2_pn = nn.Linear(self.hidden, int(self.hidden/2)).cuda()
        self.fr3_pn = nn.Linear(int(self.hidden/2), self.De).cuda()
        
        if self.vv_branch:
            self.fr1_vv = nn.Linear(2 * self.S + self.Dr, self.hidden).cuda()
            self.fr2_vv = nn.Linear(self.hidden, int(self.hidden/2)).cuda()
            self.fr3_vv = nn.Linear(int(self.hidden/2), self.De).cuda()
        self.fo1 = nn.Linear(self.P + self.Dx + (3 * self.De), self.hidden).cuda()
        self.fo2 = nn.Linear(self.hidden, int(self.hidden/2)).cuda()
        self.fo3 = nn.Linear(int(self.hidden/2), self.Do).cuda()
        
        if self.nn_branch: 
            self.fr1_nn = nn.Linear(2 * self.T + self.Dr, self.hidden).cuda()
            self.fr2_nn = nn.Linear(self.hidden, int(self.hidden/2)).cuda()
            self.fr3_nn = nn.Linear(int(self.hidden/2), self.De).cuda()
        
        if self.vv_branch:
            self.fo1_v = nn.Linear(self.S + self.Dx + (2 * self.De), self.hidden).cuda()
            self.fo2_v = nn.Linear(self.hidden, int(self.hidden/2)).cuda()
            self.fo3_v = nn.Linear(int(self.hidden/2), self.Do).cuda()
        
        if self.nn_branch: 
            self.fo1_n = nn.Linear(self.T+ self.Dx + (2 * self.De), self.hidden).cuda()
            self.fo2_n = nn.Linear(self.hidden, int(self.hidden/2)).cuda()
            self.fo3_n = nn.Linear(int(self.hidden/2), self.Do).cuda()
        

        if self.vv_branch:
            #self.fc_1 = nn.Linear(2*self.Do, self.Do).cuda()
            #self.fc_2 = nn.Linear(self.Do, int(self.Do/2)).cuda()
            #self.fc_3 = nn.Linear(int(self.Do/2), self.n_targets).cuda()
            self.fc_fixed = nn.Linear(2*self.Do, self.n_targets).cuda()
            
        elif self.nn_branch: 
            #self.fc_1 = nn.Linear(2*self.Do, self.Do).cuda()
            #self.fc_2 = nn.Linear(self.Do, int(self.Do/2)).cuda()
            #self.fc_3 = nn.Linear(int(self.Do/2), self.n_targets).cuda()
            self.fc_fixed = nn.Linear(2*self.Do, self.n_targets).cuda()
            
        else:
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
    
    def assign_matrices_neutral(self):
        self.Rc = torch.zeros(self.N, self.Nn)
        self.Rn = torch.zeros(self.Nneu, self.Nn)
        receiver_sender_list = [i for i in itertools.product(range(self.N), range(self.Nneu))]
        for i, (k, v) in enumerate(receiver_sender_list):
            self.Rc[k, i] = 1
            self.Rn[v, i] = 1
        self.Rc = (self.Rc).cuda()
        self.Rn = (self.Rn).cuda()
    
    def assign_matrices_neutralneutral(self):
        self.Rd = torch.zeros(self.Nneu, self.Ne)
        self.Rm = torch.zeros(self.Nneu, self.Ne)
        receiver_sender_list = [i for i in itertools.product(range(self.Nneu), range(self.Nneu)) if i[0]!=i[1]]
        for i, (k, v) in enumerate(receiver_sender_list):
            self.Rd[k, i] = 1
            self.Rm[v, i] = 1
        self.Rd = (self.Rd).cuda()
        self.Rm = (self.Rm).cuda()
    
    def assign_matrices_SVSV(self):
        self.Rl = torch.zeros(self.Nv, self.Ns)
        self.Ru = torch.zeros(self.Nv, self.Ns)
        receiver_sender_list = [i for i in itertools.product(range(self.Nv), range(self.Nv)) if i[0]!=i[1]]
        for i, (l, u) in enumerate(receiver_sender_list):
            self.Rl[l, i] = 1
            self.Ru[u, i] = 1
        self.Rl = (self.Rl).cuda()
        self.Ru = (self.Ru).cuda()

    def forward(self, x, y, z):
        ###PF Candidate - PF Candidate###
        Orr = self.tmul(x, self.Rr)
        Ors = self.tmul(x, self.Rs)
        B = torch.cat([Orr, Ors], 1)
        del Orr, Ors
        ### First MLP ###
        B = torch.transpose(B, 1, 2).contiguous()
        B = nn.functional.relu(self.fr1(B.view(-1, 2 * self.P + self.Dr)))
        B = nn.functional.relu(self.fr2(B))
        E = nn.functional.relu(self.fr3(B).view(-1, self.Nr, self.De))
        del B
        E = torch.transpose(E, 1, 2).contiguous()
        Ebar_pp = self.tmul(E, torch.transpose(self.Rr, 0, 1).contiguous())
        del E
        
        ####Secondary Vertex - PF Candidate### 
        Ork = self.tmul(x, self.Rk)
        Orv = self.tmul(y, self.Rv)
        B = torch.cat([Ork, Orv], 1)
        del Ork, Orv
        ### First MLP ###
        B = torch.transpose(B, 1, 2).contiguous()
        B = nn.functional.relu(self.fr1_pv(B.view(-1, self.S + self.P + self.Dr)))
        B = nn.functional.relu(self.fr2_pv(B))
        E = nn.functional.relu(self.fr3_pv(B).view(-1, self.Nt, self.De))
        del B
        E = torch.transpose(E, 1, 2).contiguous()
        Ebar_pv = self.tmul(E, torch.transpose(self.Rk, 0, 1).contiguous())
        Ebar_vp = self.tmul(E, torch.transpose(self.Rv, 0, 1).contiguous())
        del E

        ###Secondary vertex - secondary vertex###
        if self.vv_branch:
            Orl = self.tmul(y, self.Rl)
            Oru = self.tmul(y, self.Ru)
            B = torch.cat([Orl, Oru], 1)
            del Orl, Oru
            ### First MLP ###
            B = torch.transpose(B, 1, 2).contiguous()
            B = nn.functional.relu(self.fr1_vv(B.view(-1, 2 * self.S + self.Dr)))
            B = nn.functional.relu(self.fr2_vv(B))
            E = nn.functional.relu(self.fr3_vv(B).view(-1, self.Ns, self.De))
            del B
            E = torch.transpose(E, 1, 2).contiguous()
            Ebar_vv = self.tmul(E, torch.transpose(self.Rl, 0, 1).contiguous())
            del E
            
        if self.nn_branch: 
            Ord = self.tmul(z, self.Rd)
            Orm = self.tmul(z, self.Rm)
            B = torch.cat([Ord, Orm], 1)
            del Ord, Orm
            ### First MLP ###
            B = torch.transpose(B, 1, 2).contiguous()
            B = nn.functional.relu(self.fr1_nn(B.view(-1, 2 * self.T + self.Dr)))
            B = nn.functional.relu(self.fr2_nn(B))
            E = nn.functional.relu(self.fr3_nn(B).view(-1, self.Ne, self.De))
            del B
            E = torch.transpose(E, 1, 2).contiguous()
            Ebar_nn = self.tmul(E, torch.transpose(self.Rd, 0, 1).contiguous())
            del E
        ### PF candidate - neutral particle ###

        Orc = self.tmul(x, self.Rc)
        Orn = self.tmul(z, self.Rn)
        B = torch.cat([Orc, Orn], 1)
        del Orc, Orn
        ### First MLP ###
        B = torch.transpose(B, 1, 2).contiguous()
        B = nn.functional.relu(self.fr1_pn(B.view(-1, self.T + self.P + self.Dr)))
        B = nn.functional.relu(self.fr2_pn(B))
        E = nn.functional.relu(self.fr3_pn(B).view(-1, self.Nn, self.De))
        del B
        E = torch.transpose(E, 1, 2).contiguous()
        Ebar_pn = self.tmul(E, torch.transpose(self.Rc, 0, 1).contiguous())
        Ebar_np = self.tmul(E, torch.transpose(self.Rn, 0, 1).contiguous())
        del E
        
        ####Final output matrix for particles###
        C = torch.cat([x, Ebar_pp, Ebar_pv, Ebar_pn], 1)
        del Ebar_pp
        del Ebar_pv
        del Ebar_pn
    
        if not self.vv_branch:
            del Ebar_vp
            
        if not self.nn_branch:
            del Ebar_np    
    
        C = torch.transpose(C, 1, 2).contiguous()
        ### Second MLP ###
        C = nn.functional.relu(self.fo1(C.view(-1, self.P + self.Dx + (3 * self.De))))
        C = nn.functional.relu(self.fo2(C))
        O = nn.functional.relu(self.fo3(C).view(-1, self.N, self.Do))
        del C

        if self.vv_branch:
            ####Final output matrix for particles### 
            C = torch.cat([y, Ebar_vv, Ebar_vp], 1)
            del Ebar_vv
            del Ebar_vp
            C = torch.transpose(C, 1, 2).contiguous()
            ### Second MLP ###
            C = nn.functional.relu(self.fo1_v(C.view(-1, self.S + self.Dx + (2 * self.De))))
            C = nn.functional.relu(self.fo2_v(C))
            O_v = nn.functional.relu(self.fo3_v(C).view(-1, self.Nv, self.Do))
            del C
        
        if self.nn_branch:
            ####Final output matrix for particles### 
            C = torch.cat([z, Ebar_nn, Ebar_np], 1)
            del Ebar_nn
            del Ebar_np
            C = torch.transpose(C, 1, 2).contiguous()
            ### Second MLP ###
            C = nn.functional.relu(self.fo1_n(C.view(-1, self.T + self.Dx + (2 * self.De))))
            C = nn.functional.relu(self.fo2_n(C))
            O_n = nn.functional.relu(self.fo3_n(C).view(-1, self.Nneu, self.Do))
            del C
        
        #Taking the sum of over each particle/vertex
        N = torch.sum(O, dim=1)
        del O
        if self.vv_branch:
            N_v = torch.sum(O_v,dim=1)
            del O_v
        if self.nn_branch: 
            N_n = torch.sum(O_n,dim=1)
            del O_n
        
        ### Classification MLP ###
        if self.vv_branch:
            #N = nn.functional.relu(self.fc_1(torch.cat([N, N_v],1)))
            #N = nn.functional.relu(self.fc_2(N))
            #N = self.fc_3(N)
            N =self.fc_fixed(torch.cat([N, N_v],1))
        
        elif self.nn_branch: 
            #N = nn.functional.relu(self.fc_1(torch.cat([N, N_n],1)))
            #N = nn.functional.relu(self.fc_2(N))
            #N = self.fc_3(N)
            N =self.fc_fixed(torch.cat([N, N_n],1))
        else:
            N = self.fc_fixed(N)

        return N 
            
    def tmul(self, x, y):  #Takes (I * J * K)(K * L) -> I * J * L 
        x_shape = x.size()
        y_shape = y.size()
        return torch.mm(x.view(-1, x_shape[2]), y).view(-1, x_shape[1], y_shape[1])
    
    
class Rx(nn.Module):
    def __init__(self,Do=6, hidden=64, nbins=40 ):
        super(Rx, self).__init__()
        self.dense1 = nn.Linear(Do, hidden).cuda()
        self.dense2 = nn.Linear(hidden, hidden).cuda()
        self.dense3 = nn.Linear(hidden, hidden).cuda()
        self.dense4 = nn.Linear(hidden, nbins).cuda()
        
    def forward(self, x):
        n = nn.functional.relu(self.dense1(x))
        n = nn.functional.relu(self.dense2(n))
        n = nn.functional.relu(self.dense3(n))
        n = self.dense4(n)
        return n
