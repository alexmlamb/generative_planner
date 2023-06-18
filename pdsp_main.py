

'''
First get a pre-trained dynamics model.  
Then start by training 1-step inverse model.  

Then do a k-step inverse model using traj from dataset.  

1 - Start collecting s,a data.  
2 - Train 1-step inverse model p(a | s[t], s[t+1]) using fourier features.  


'''

import torch
import torch.nn as nn
from dyn_model import DynVAE, GaussianFourierProjectionTime
from utils import sample_batch
import pickle
import random
import numpy

import time

class Planner(nn.Module):

    def __init__(self):
        super().__init__()

        #self.trans = nn.Sequential(nn.Linear(2+2, 1024), nn.Dropout(0.5), nn.LeakyReLU(), nn.Linear(1024,1024), nn.Dropout(0.5), nn.LeakyReLU(), nn.Linear(1024,1024), nn.Dropout(0.5), nn.LeakyReLU(), nn.Linear(1024,4))

        self.gauss_feat = GaussianFourierProjectionTime().cuda()

        self.net = nn.Sequential(nn.Linear(512*4 + 0, 1024), nn.LeakyReLU(), nn.Linear(1024,1024), nn.LeakyReLU(), nn.Linear(1024,1024), nn.LeakyReLU(), nn.Linear(1024, 2))

        self.koptflag = nn.Embedding(2, 512)

        self.lfn = nn.MSELoss()

    def makefeat(self,st,stk):

        bs = st.shape[0]
        inp = torch.cat([st,stk],dim=1)
        inp_flat = inp.reshape((inp.shape[0]*inp.shape[1]))
        feat = self.gauss_feat(inp_flat).reshape((bs,-1))

        return feat

    def makekemb(self, k, bs):
        k = torch.Tensor([k]*bs).long().cuda()
        kemb = self.koptflag(k)
        return kemb

    def score_traj(self, slst, alst, sg):
        #for t in range(len(slst)):

        score = 0.0
        lamb = 0.95
        max_score = 0.0

        #print('scoring traj!')
        #print('s-shape', slst[0].shape)
        #print('sg shape', sg.shape)
        #print('slen', len(slst), 'alen', len(alst))
        #raise Exception('done')

        dyn_hit = 1.0


        slast = slst[:-1]
        snext = slst[1:]

        T, bs, ns = slast.shape
        _, _, na = alst.shape

        sg = sg.unsqueeze(0).repeat(T,1,1).reshape((T*bs,-1))

        slast = slast.reshape((T*bs,-1))
        snext = snext.reshape((T*bs,-1))
        alst = alst.reshape((T*bs,-1))

        #print(slast.shape, snext.shape, alst.shape, sg.shape)

        c_dyn_score = self.score.forward_enc(slast, alst, snext, 1)[1]
        c_dyn_score = torch.clamp(c_dyn_score, 0.0, 0.9)/0.9

        c_goal_score = self.score.forward_enc(slast, alst, sg, 1)[1]

        c_dyn_score = c_dyn_score.reshape((T,bs))
        c_goal_score = c_goal_score.reshape((T,bs))

        c_dyn_score = c_dyn_score.prod(dim=0)
        c_goal_score = c_goal_score.mean(dim=0)

        score = c_dyn_score * c_goal_score

        #score = c_dyn_score * c_goal_score

        #score = score.reshape((T, bs))

        #score = score.mean(dim=0)

        return score

        #for j in range(len(slst)-1):
        #    weighting = lamb**(len(slst)-1-j)
        #    c_dyn_score = self.score.forward_enc(slst[j], alst[j], slst[j+1], 1)[1]
        #    c_goal_score = self.score.forward_enc(slst[j], alst[j], sg, 1)[1]
        #    score += weighting * c_goal_score
        #    dyn_hit *= torch.clamp(c_dyn_score, 0.0, 0.9) / 0.9
        #    max_score += weighting
        #score *= dyn_hit
        #return score / max_score

    def simulate(self, s1, sg, dyn, ksim):

        slst = []
        alst = []

        s = s1*1.0

        for j in range(0,ksim):
            a = self.net(self.makefeat(s,sg))
            s = dyn(s, a)*1.0
            #s = dyn(s.round(decimals=2), a.round(decimals=2)).round(decimals=2)*1.0
            slst.append(s.unsqueeze(0))
            alst.append(a.unsqueeze(0))

        #print('s shape', s.shape, 'a shape', a.shape)

        slst = torch.cat(slst, dim=0)
        alst = torch.cat(alst, dim=0)

        return slst, alst

    def multistep(self,s0,a0,sg,ktr,dyn):

        '''
            Compute first action under net, and randomly.  
            For both, take 1-step under the dynamics.  
            Then compute next action under net, then take step under dynamics
            For both sequences, compute reward for reaching goal sg.  To start with, reward could just be e^(-(s1-sg)**2)
            Figure out which is better, then use that a0 as target to update net.  
            
        '''

        #RT_a0 = (torch.rand_like(a0)-0.5)*(2.0/5.0)

        OT_a0 = torch.clamp(self.net(self.makefeat(s0,sg)),-0.2,0.2)
        OT_s1 = dyn(s0, OT_a0)

        RT_a0 = torch.clamp(OT_a0+torch.randn_like(OT_a0)*0.1, -0.2, 0.2)
        RT_s1 = dyn(s0, RT_a0)

        #RT_a1 = self.net(self.makefeat(RT_s1,sg))
        #RT_s2 = dyn(RT_s1, RT_a1)

        t0 = time.time()

        T_s1 = torch.cat([RT_s1, OT_s1], dim=0)
        T_sg = torch.cat([sg, sg], dim=0)


        T_slst, T_alst = self.simulate(T_s1, T_sg, dyn, ktr-1)

        RT_slst, OT_slst = torch.chunk(T_slst, 2, dim=1)
        RT_alst, OT_alst = torch.chunk(T_alst, 2, dim=1)

        RT_slst = torch.cat([s0.unsqueeze(0), RT_s1.unsqueeze(0), RT_slst], dim=0)
        OT_slst = torch.cat([s0.unsqueeze(0), OT_s1.unsqueeze(0), OT_slst], dim=0)

        RT_alst = torch.cat([RT_a0.unsqueeze(0), RT_alst], dim=0)
        OT_alst = torch.cat([OT_a0.unsqueeze(0), OT_alst], dim=0)

        assert RT_slst.shape == OT_slst.shape
        assert RT_alst.shape == OT_alst.shape

        T_slst = torch.cat([RT_slst, OT_slst], dim=1)
        T_alst = torch.cat([RT_alst, OT_alst], dim=1)
        T_sg = torch.cat([sg,sg],dim=0)

        T_score = self.score_traj(T_slst, T_alst, T_sg)

        RT_score, OT_score = torch.chunk(T_score, 2, dim = 0)


        if random.uniform(0,1) < 0.001:
            print("RT-score", RT_score[0:10])
            print("OT-score", OT_score[0:10])

        sel_RT = torch.gt(RT_score, OT_score).float().unsqueeze(1)

        a0_targ = sel_RT * RT_a0 + (1-sel_RT) * OT_a0
        sk_targ = sel_RT * RT_slst[-1] + (1-sel_RT) * OT_slst[-1]

        a0_pred = self.net(self.makefeat(s0.detach(),sg.detach())) #Better to use sg (goal) or what we reached (sk_targ).  

        l = self.lfn(a0_pred, a0_targ.detach())

        return l, a0_pred, torch.cat([s0.unsqueeze(0), OT_slst], dim=0), OT_score

    def loss(self,st,at,stk):
        bs = st.shape[0]

        feat = self.makefeat(st,stk)


        out = self.net(feat)

        l = self.lfn(out, at.detach())

        return l, out, [st,stk], None

if __name__ == "__main__":

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    dyn = torch.load('dyn.pt').cuda() #takes sl, al and returns sn
    score = torch.load('contrastive.pt').cuda()  #takes sl,al,sn and returns score

    dataset = pickle.load(open('data/dataset.p', 'rb'))
    X, A, ast, est = dataset['X'], dataset['A'], dataset['ast'], dataset['est']

    plan = Planner().cuda()

    plan.score = score

    opt = torch.optim.Adam(plan.parameters(), lr=0.0001)

    loss_lst = []

    for j in range(0,300000):

        datak = 32
        st, a = sample_batch(X, A, ast, est, 128, datak) #bs x pos x 2


        ktr = random.choice([1,15])#random.randint(1,31)


        if ktr == 1:
            #1-step problem
            s0 = st[:,0]
            sk = st[:,1]
            a0 = a[:,0]

            l,a0pred,splan, plan_score = plan.loss(s0,a0,sk)
        else:
            #t0 = time.time()
            s0 = st[:,0]
            sk = s0[torch.randperm(s0.shape[0])]#st[:,ktr]
            a0 = a[:,0]
            l,a0pred,splan, plan_score = plan.multistep(s0,a0,sk,ktr,dyn)
            #print(time.time() - t0, 'total forward time')

        loss_lst.append(l.item())

        l.backward()
        opt.step()
        opt.zero_grad()


        if j % 2000 == 0:
            print(j, sum(loss_lst)/len(loss_lst))
            loss_lst = []
            print(ktr)
            print(s0[0],sk[0],a0pred[0])


            if ktr != 1:

                planlst = list(map(lambda sp: list(map(lambda z: round(z, 2), sp[0].cpu().data.numpy().tolist())), splan))
                print("Plan", list(map(lambda sp: list(map(lambda z: round(z, 2), sp[0].cpu().data.numpy().tolist())), splan)))
                print("Plan score", plan_score[0])

                plt.plot([0.5,0.5], [0.0,0.4], color='black', linewidth=3)
                plt.plot([0.5,0.5], [0.6,1.0], color='black', linewidth=3)
                plt.plot([0.2,0.8], [0.6,0.6], color='black', linewidth=3)
                plt.plot([0.2,0.8], [0.4,0.4], color='black', linewidth=3)

                plt.plot(numpy.array(planlst)[:,0], numpy.array(planlst)[:,1], linewidth=3, **{'color': 'lightsteelblue', 'marker': 'o'}, zorder=1)

                plt.scatter(x=s0[0][0].item(), y=s0[0][1].item(), color='blue',zorder=2)
                plt.scatter(x=sk[0][0].item(), y=sk[0][1].item(), color='green',zorder=3)

                plt.xlim([0.0,1.0])
                plt.ylim([0.0,1.0])

                plt.savefig('results/plan.png')
                plt.clf()





