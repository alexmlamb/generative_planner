

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

from viz_utils import viz_plan

amax = 0.2

class PolicyValueNet(nn.Module):

    def __init__(self):
        super().__init__()

        #(s,g) --> a
        self.policy_net = nn.Sequential(nn.Linear(512*4, 1024), nn.LeakyReLU(), nn.Linear(1024,1024), nn.LeakyReLU(), nn.Linear(1024,2)).cuda() 

        #(s,g) --> 1
        self.value_net = nn.Sequential(nn.Linear(512*4, 1024), nn.LeakyReLU(), nn.Linear(1024,1024), nn.LeakyReLU(), nn.Linear(1024,1)).cuda()

        #(s,g,a) --> 1
        self.q_net = nn.Sequential(nn.Linear(512*6, 1024), nn.LeakyReLU(), nn.Linear(1024,1024), nn.LeakyReLU(), nn.Linear(1024,1)).cuda()

        self.gauss_feat = GaussianFourierProjectionTime()

    def policy(self, st, sg):
        st = self.makefeat2(st)
        sg = self.makefeat2(sg)
        feat = torch.cat([st,sg],dim=1)

        p = self.policy_net(feat)

        return p

    def makefeat2(self,s):
        bs = s.shape[0]
        inp_flat = s.reshape((bs*s.shape[1]))
        feat = self.gauss_feat(inp_flat).reshape((bs,-1))

        return feat

    def value(self, st, sg):
        st = self.makefeat2(st)
        sg = self.makefeat2(sg)
        feat = torch.cat([st,sg],dim=1)

        v = self.value_net(feat)

        return v

    def qval(self, st, at, sg):
        st = self.makefeat2(st)
        at = self.makefeat2(at)
        sg = self.makefeat2(sg)
        feat = torch.cat([st,at,sg], dim=1)
        q = self.q_net(feat)
        return q



class Planner(nn.Module):

    def __init__(self):
        super().__init__()

        #self.trans = nn.Sequential(nn.Linear(2+2, 1024), nn.Dropout(0.5), nn.LeakyReLU(), nn.Linear(1024,1024), nn.Dropout(0.5), nn.LeakyReLU(), nn.Linear(1024,1024), nn.Dropout(0.5), nn.LeakyReLU(), nn.Linear(1024,4))


        #self.pvn = nn.Sequential(nn.Linear(512*4 + 512, 1024), nn.LeakyReLU(), nn.Linear(1024,1024), nn.LeakyReLU(), nn.Linear(1024,1024), nn.LeakyReLU(), nn.Linear(1024, 2))
        self.koptflag = nn.Embedding(2, 512)

        self.pvn = PolicyValueNet().cuda()

        self.lfn = nn.MSELoss(reduction='none')

    def makefeat(self,s):

        bs = s.shape[0]
        inp = s#torch.cat([st,stk],dim=1)
        inp_flat = inp.reshape((inp.shape[0]*inp.shape[1]))
        feat = self.gauss_feat(inp_flat).reshape((bs,-1))

        return feat

    def makekemb(self, k, bs):
        k = torch.Tensor([k]*bs).long().cuda()
        kemb = self.koptflag(k)
        return kemb

    def score_traj(self, slst, alst, sg):

        score = 0.0
        dyn_hit = 1.0

        slast = slst[:-1]
        snext = slst[1:]

        T, bs, ns = slast.shape
        _, _, na = alst.shape

        sg = sg.unsqueeze(0).repeat(T,1,1).reshape((T*bs,-1))

        slast = slast.reshape((T*bs,-1))
        snext = snext.reshape((T*bs,-1))
        alst = alst.reshape((T*bs,-1))

        c_dyn_score = self.score.forward_enc(slast, alst, snext, 1)[1]
        c_dyn_score = torch.clamp(c_dyn_score, 0.0, 0.9)/0.9

        #k = torch.Tensor([0]*slast.shape[0]).long().cuda()
        #kemb = self.koptflag(k)
        #_, val_est = self.pvn(torch.cat([self.makefeat(slast, sg), kemb], dim=1))
        
        train_score = 0
        eval_score = 0

        #train_score += self.score.forward_enc(slast, alst, sg, 0)[1]*0.25
        
        train_score += torch.exp(-100.0 * torch.abs(slast - sg).mean(dim=1)) * 0.25
        train_score += self.score.forward_enc(slast, alst, sg, 1)[1]*0.25
        train_score += self.score.forward_enc(slast, alst, sg, 2)[1]*0.50

        eval_score += self.score.forward_enc(slast, alst, sg, 2)[1] #michael henaff, yann lecun.  Uncertainty-based penalization of model-based RL.  
        #curiosity driven exploration

        c_dyn_score = c_dyn_score.reshape((T,bs))
        train_score = train_score.reshape((T,bs))
        eval_score = eval_score.reshape((T,bs))

        #val_est = val_est.mean(dim=1)
        #val_est = val_est.reshape((T,bs))

        c_dyn_score = c_dyn_score.cumprod(dim=0)
        
        lamb = train_score*0.0 + 0.95
        weighting = torch.cumprod(lamb, dim=0)

        train_score = c_dyn_score * train_score * weighting
        eval_score = c_dyn_score * eval_score

        return train_score, eval_score#, val_est

    def simulate(self, s1, sg, sg_targ, dyn, ksim):

        slst = []
        alst = []

        s = s1*1.0

        k = torch.Tensor([0]*s1.shape[0]).long().cuda()
        kemb = self.koptflag(k)

        for j in range(0,ksim):

            if j < ksim - 2:
                sg_use = sg_targ
            else:
                sg_use = sg

            a,_ = self.pvn(torch.cat([self.makefeat(s,sg_use), kemb], dim=1))
            s = dyn(s, a)*1.0
            slst.append(s.unsqueeze(0))
            alst.append(a.unsqueeze(0))

        #print('s shape', s.shape, 'a shape', a.shape)

        slst = torch.cat(slst, dim=0)
        alst = torch.cat(alst, dim=0)

        return slst, alst

    def multistep(self,s0,a0,sg,sg_pre,ktr,dyn):

        '''
            Compute first action under net, and randomly.  
            For both, take 1-step under the dynamics.  
            Then compute next action under net, then take step under dynamics
            For both sequences, compute reward for reaching goal sg.  To start with, reward could just be e^(-(s1-sg)**2)
            Figure out which is better, then use that a0 as target to update net.  
            
        '''

        #RT_a0 = (torch.rand_like(a0)-0.5)*(2.0/5.0)

        k = torch.Tensor([0]*s0.shape[0]).long().cuda()
        kemb = self.koptflag(k)

        if ktr > 2 and random.uniform(0,1) < 0.5:
            sg_targ = sg_pre
            noise_level = 0.0
        else:
            sg_targ = sg
            noise_level = random.uniform(0.1, 0.2)

        OT_a0 = torch.clamp(self.pvn(torch.cat([self.makefeat(s0,sg), kemb], dim=1))[0],-1*amax,amax)
        OT_s1 = dyn(s0, OT_a0)

        RT_a0 = torch.clamp(self.pvn(torch.cat([self.makefeat(s0,sg_targ), kemb], dim=1))[0]+torch.randn_like(OT_a0)*0.1, -0.2, 0.2)
        RT_s1 = dyn(s0, RT_a0)


        t0 = time.time()

        T_s1 = torch.cat([RT_s1, OT_s1], dim=0)
        T_sg = torch.cat([sg, sg], dim=0)
        T_sg_targ = torch.cat([sg_targ, sg], dim=0)

        T_slst, T_alst = self.simulate(T_s1, T_sg, T_sg_targ, dyn, ktr-1)

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

        T_score_weighted, T_score = self.score_traj(T_slst, T_alst, T_sg)

        RT_score, OT_score = torch.chunk(T_score, 2, dim = 1)
        RT_score_weighted, OT_score_weighted = torch.chunk(T_score_weighted, 2, dim = 1)
        #RT_val, OT_val = torch.chunk(val_est, 2, dim=1)

        R_score = RT_score_weighted.mean(dim=0)# + RT_val.mean(dim=0)
        O_score = OT_score_weighted.mean(dim=0)# + OT_val.mean(dim=0)

        if random.uniform(0,1) < 0.001:
            print("RT-score", R_score[0:10])
            print("OT-score", O_score[0:10])

        sel_RT = torch.gt(R_score, O_score).float().unsqueeze(1)

        #print('sel-shape', sel_RT.shape)

        a0_targ = sel_RT * RT_a0 + (1-sel_RT) * OT_a0
        sk_targ = sel_RT * RT_slst[-1] + (1-sel_RT) * OT_slst[-1]

        a0_pred, v_pred = self.pvn(torch.cat([self.makefeat(s0.detach(),sg.detach()), kemb], dim=1)) #Better to use sg (goal) or what we reached (sk_targ).  
            
        l = self.lfn(a0_pred, a0_targ.detach()).mean()

        #l += self.lfn(v_pred, OT_score_weighted.permute(1,0).detach()).mean()

        return l, a0_pred, torch.cat([s0.unsqueeze(0), OT_slst], dim=0), OT_score, v_pred

    def onestep_loss(self, st, at, stk):

        feat = self.makefeat(st,stk)

        k = torch.Tensor([0]*st.shape[0]).long().cuda()
        kemb = self.koptflag(k)

        a0_OT = self.pvn(torch.cat([feat,kemb],dim=1))[0]

        if False:
            a0_RT = at
        else:
            a0_RT = torch.clamp(a0_OT.detach()+torch.randn_like(a0_OT)*0.01, -1*amax, amax)

        score_OT = -1.0 * ((self.dyn(st, a0_OT.detach()) - stk)**2).mean(dim=1) - (a0_OT**2).mean(dim=1) * 0.01
        score_RT = -1.0 * ((self.dyn(st, a0_RT) - stk)**2).mean(dim=1) - (a0_RT**2).mean(dim=1) * 0.01

        sel_RT = torch.gt(score_RT, score_OT).float().unsqueeze(1)

        a0_pred = sel_RT * a0_RT + (1-sel_RT)*a0_OT

        l = self.lfn(a0_OT, a0_pred.detach()).mean()

        return l

    def cguard(self, st, at, sn):
        c = self.score.forward_enc(st, at, sn, 1)[1]
        c = c.unsqueeze(1)
        return c

    def reward(self, st, at, sg):
        r = 0.0

        if False:
            r += self.score.self_score(st, sg)[1]*1.0
        else:
            r += torch.exp(-100.0 * torch.abs(st - sg).mean(dim=1)) * 0.0
            r += self.score.forward_enc(st, at, sg, 1)[1]*0.5
            r += self.score.forward_enc(st, at, sg, 2)[1]*0.5
        r = r.unsqueeze(1)
        return r

    def eval_value(self, sg):
        #sg is (1x2)
        s0 = torch.rand((2048,2)).cuda()
        s0 = torch.cat([s0, sg], dim=0)
        sg = sg.repeat(2048+1,1)

        vals = self.pvn.value(s0,sg)

        opt_val = vals.argmax(dim=0)
        opt_s = s0[opt_val]

        policy = self.pvn.policy(s0,sg)

        return vals.cpu().data, s0.cpu().data, sg[0:1].cpu().data, opt_s.cpu().data, policy.cpu().data

    def loss(self,st,sg):
        bs = st.shape[0]
        N = 5
        l = 0.0
        mse = nn.MSELoss()

        #First, estimate Q(s,a,g) := R(s,a,g).  
        #Get set of random actions (bs,N,2)
        #Make (s,g) repeat 

        k = torch.randint(1,15,size=(bs,1)).repeat(1,N).cuda()

        policy = self.pvn.policy(st, sg).unsqueeze(1)

        a = torch.clamp(torch.randn(bs,N-1,2)*0.1, -0.2, 0.2).cuda()
        a = torch.cat([a,policy],dim=1)


        a[:,0:1,:] *= 0.0

        st_rep = st.unsqueeze(1).repeat(1,N,1).cuda()
        sg_rep = sg.unsqueeze(1).repeat(1,N,1).cuda()
        
        st_rep = st_rep.reshape((bs*N,-1))
        a_rep = a.reshape((bs*N,-1))
        sg_rep = sg_rep.reshape((bs*N,-1))

        Q_val = self.pvn.qval(st_rep, a_rep, sg_rep)
        r = self.reward(st_rep, a_rep, sg_rep)
        
        sn_rep = self.dyn(st_rep, a_rep)
        QV_targ = self.pvn.value(sn_rep,sg_rep)

        Q_targ = r + (0.95 * QV_targ)
        cg = self.cguard(st_rep, a_rep, sn_rep)
        
        Q_targ *= cg * torch.gt(cg, 0.9).float()

        l += mse(Q_val, Q_targ.detach())

        #Now estimate V(s,g) := max_a Q(s,a,g)
        Q_val = Q_val.reshape((bs,N,1))
        maxQ = Q_val.max(dim=1).values.detach()

        policy_target = a_rep.reshape((bs,N,2))[torch.arange(128), Q_val.argmax(dim=1).squeeze(-1), :]
        policy = self.pvn.policy(st, sg)
        l += mse(policy, policy_target.detach())

        V = self.pvn.value(st,sg) #estimate at k
        l += mse(V, maxQ)

        return l

if __name__ == "__main__":

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    dyn = torch.load('dyn.pt').cuda() #takes sl, al and returns sn
    score = torch.load('contrastive.pt').cuda()  #takes sl,al,sn and returns score

    dataset = pickle.load(open('data/dataset.p', 'rb'))
    X, A, ast, est = dataset['X'], dataset['A'], dataset['ast'], dataset['est']

    plan = Planner().cuda()

    opt = torch.optim.Adam(plan.parameters(), lr=0.0001)

    plan.score = score
    plan.dyn = dyn


    loss_lst = []
    score_lst = []
    score_approx_lst = []

    for j in range(0,2*300000):

        datak = 32
        st, a = sample_batch(X, A, ast, est, 128, datak) #bs x pos x 2

        if True:
            s0 = st[:,0]
            idx = torch.randperm(s0.shape[0])
            sk = st[:,-1][idx]
            sk_pre = st[:,-2][idx]
            a0 = a[:,0]

            l = plan.loss(s0,sk)

            loss_lst.append(l.item())

        l.backward()
        opt.step()
        opt.zero_grad()

        if j % 2000 == 0:
            print(j, sum(loss_lst)/len(loss_lst))
            loss_lst = []

            vals, s0, sg, opt_s, policy = plan.eval_value(sk[0:1].cuda())
            plt.scatter(s0[:,0], s0[:,1], c=vals, cmap='inferno')
            plt.savefig('results_val/vals.png')
            plt.clf()

            policy = torch.clamp(policy, -0.05, 0.05)

            plt.quiver(s0[:,0], s0[:,1], policy[:,0], policy[:,1])
            plt.savefig('results_val/policy.png')
            plt.clf()

            print('v min max', vals.min(), vals.max())        
            print('sg, opt_s', sg, opt_s)






