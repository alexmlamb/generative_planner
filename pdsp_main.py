

import torch
import torch.nn as nn
from dyn_model import DynVAE, GaussianFourierProjectionTime
from utils import sample_batch
import pickle
import random
import numpy
import torch.nn.functional as F

import time

from viz_utils import viz_plan, viz_plan_diverse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

amax = 0.2

class StochasticPolicy(nn.Module):
    def __init__(self):
        super().__init__()

        h_dim = 512*4 + 512
        self.a_feat = nn.Linear(2, 1024)

        self.prior_net = nn.Sequential(nn.Linear(h_dim, 1024), nn.LeakyReLU(), nn.Linear(1024, 128))
        self.posterior_net = nn.Sequential(nn.Linear(h_dim + 512*2, 1024), nn.LeakyReLU(), nn.Linear(1024, 128))
        self.decoder_net = nn.Sequential(nn.Linear(h_dim + 64, 1024), nn.LeakyReLU(), nn.Linear(1024,1024), nn.LeakyReLU(), nn.Linear(1024,1024), nn.LeakyReLU(), nn.Linear(1024,1024), nn.LeakyReLU(), nn.Linear(1024,2))

        self.gauss_feat = GaussianFourierProjectionTime().cuda()

    def mode(self, h):
        pr_h = self.prior_net(h)
        mu,std = torch.chunk(pr_h,2,dim=1)
        std = torch.exp(std)

        prior = torch.distributions.normal.Normal(mu, std)
        z_prior = prior.rsample()

        out = self.decoder_net(torch.cat([h,z_prior], dim=1))
        return out

    def train_act(self, h, a):

        af = self.a_feat(a)
        post_h = self.posterior_net(torch.cat([h, af], dim=1))
        h_mu, h_std = torch.chunk(post_h, 2, dim=1)
        h_std = torch.exp(h_std)

        h_prior = self.prior_net(h)
        mu_prior, std_prior = torch.chunk(h_prior, 2, dim=1)
        std_prior = torch.exp(std_prior)

        prior = torch.distributions.normal.Normal(mu_prior, std_prior)
        posterior = torch.distributions.normal.Normal(h_mu, h_std)
        kl_loss = torch.distributions.kl_divergence(posterior, prior).sum(dim=-1).mean() * 1e-5
        z_post = posterior.rsample()

        out = self.decoder_net(torch.cat([h,z_post], dim=1))

        return out, kl_loss


class DeterministicPolicy(nn.Module):
    def __init__(self):
        super().__init__()

        h_dim = 512*4 + 512
        self.a_feat = nn.Linear(2, 1024)

        self.decoder_net = nn.Sequential(nn.Linear(h_dim, 1024), nn.LeakyReLU(), nn.Linear(1024,1024), nn.LeakyReLU(), nn.Linear(1024,1024), nn.LeakyReLU(), nn.Linear(1024,1024), nn.LeakyReLU(), nn.Linear(1024,2))

        self.value_net = nn.Sequential(nn.Linear(512*6, 1024), nn.LeakyReLU(), nn.Linear(1024,1024), nn.LeakyReLU(), nn.Linear(1024, 1)) #(s,a)
        self.gauss_feat = GaussianFourierProjectionTime().cuda()


    def mode(self, h):
        out = self.decoder_net(torch.cat([h], dim=1))
        return out


    def value(self, s, a, sg):
        bs = s.shape[0]
        inp = torch.cat([s,a,sg],dim=1)
        inp_flat = inp.reshape((inp.shape[0]*inp.shape[1]))
        feat = self.gauss_feat(inp_flat).reshape((bs,-1))
        h = self.value_net(feat)
        return h

    def train_act(self, h, a):


        out = self.decoder_net(torch.cat([h], dim=1))

        return out

class PolicyValueNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.policy_net = nn.Sequential(nn.Linear(512*4 + 512, 1024), nn.LeakyReLU(), nn.Linear(1024,1024), nn.LeakyReLU(), nn.Linear(1024,1024), nn.LeakyReLU())
        self.value_net = nn.Sequential(nn.Linear(512*4 + 512, 1024), nn.LeakyReLU(), nn.Linear(1024,1024), nn.LeakyReLU(), nn.Linear(1024,1024), nn.LeakyReLU())

        self.policy_head = nn.Linear(1024,2)
        self.value_head = nn.Linear(1024,15)


    def forward(self, h):

        p = self.policy_head(self.policy_net(h))
        v = self.value_head(self.value_net(h))

        return p,v


class Planner(nn.Module):

    def __init__(self):
        super().__init__()

        self.gauss_feat = GaussianFourierProjectionTime().cuda()

        self.koptflag = nn.Embedding(2, 512)

        self.pvn = DeterministicPolicy().cuda()
        self.diverse_pvn = StochasticPolicy().cuda()

        self.lfn = nn.MSELoss(reduction='none')

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

    def eval_value(self, sg):

        X,Y = torch.meshgrid(torch.arange(0.0,1.0,0.03), torch.arange(0.0,1.0,0.03), indexing='ij')
        s0 = torch.cat([X.unsqueeze(-1), Y.unsqueeze(-1)], dim=2).reshape((X.shape[0]*X.shape[1], 2)).cuda()

        s0 = torch.cat([s0, sg], dim=0)
        sg = sg.repeat(s0.shape[0],1)

        a0 = s0*0.0

        vals = self.pvn.value(s0,a0,sg)

        opt_val = vals.argmax(dim=0)
        opt_s = s0[opt_val]

        k = torch.Tensor([0]*s0.shape[0]).long().cuda()
        kemb = self.koptflag(k)
        a = self.pvn.mode(torch.cat([self.makefeat(s0,sg), kemb], dim=1))

        return vals.cpu().data, s0.cpu().data, sg[0:1].cpu().data, opt_s.cpu().data, a.cpu().data

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

        train_score = 0
        eval_score = 0

        if True:
            train_score += self.score.self_score(slast, sg)[1]*0.333
            train_score += self.score.forward_enc(slast, alst, sg, 1)[1]*0.333
            train_score += self.score.forward_enc(slast, alst, sg, 2)[1]*0.333
        else:
            train_score += torch.exp(-100.0 * torch.abs(slast - sg).mean(dim=1)) * 1.0
            train_score += self.score.forward_enc(slast, alst, sg, 1)[1]*0.0
            train_score += self.score.forward_enc(slast, alst, sg, 2)[1]*0.0

        eval_score += self.score.forward_enc(slast, alst, sg, 2)[1] #michael henaff, yann lecun.  Uncertainty-based penalization of model-based RL.  
        #curiosity driven exploration

        c_dyn_score = c_dyn_score.reshape((T,bs))
        train_score = train_score.reshape((T,bs))
        eval_score = eval_score.reshape((T,bs))


        c_dyn_score = c_dyn_score.cumprod(dim=0)
        
        lamb = train_score*0.0 + 1.0
        weighting = torch.cumprod(lamb, dim=0)

        train_score = c_dyn_score * train_score * weighting

        eval_score = c_dyn_score * eval_score

        return train_score, eval_score#, val_est

    def simulate(self, s1, sg, sg_targ, dyn, ksim, diverse=False):

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

            if diverse:
                a = self.diverse_pvn.mode(torch.cat([self.makefeat(s,sg_use), kemb], dim=1))
            else:
                a = self.pvn.mode(torch.cat([self.makefeat(s,sg_use), kemb], dim=1))

            s = dyn(s, a)*1.0
            
            slst.append(s.unsqueeze(0))
            alst.append(a.unsqueeze(0))


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


        k = torch.Tensor([0]*s0.shape[0]).long().cuda()
        kemb = self.koptflag(k)

        bs = s0.shape[0]
        if True and ktr > 2 and random.uniform(0,1) < 0.5:
            sg_targ = sg_pre
        else:
            sg_targ = sg

        NR = 1
        T = ktr-1
        bsa = bs*(NR+1)
        sdim = s0.shape[1]
        adim = a0.shape[1]

        OT_a0 = torch.clamp(self.pvn.mode(torch.cat([self.makefeat(s0,sg), kemb], dim=1)),-1*amax,amax)
        OT_s1 = dyn(s0, OT_a0)

        RT_a0 = self.pvn.mode(torch.cat([self.makefeat(s0,sg_targ), kemb], dim=1))
        RT_a0 = RT_a0.unsqueeze(0).repeat(NR, 1, 1).reshape((NR*OT_a0.shape[0], OT_a0.shape[1]))
        RT_s0 = s0.unsqueeze(0).repeat(NR,1,1).reshape((NR*s0.shape[0], s0.shape[1]))
        RT_a0 = torch.clamp(RT_a0 + torch.randn_like(RT_a0)*0.1, -0.2, 0.2)
        RT_s1 = dyn(RT_s0, RT_a0)

        t0 = time.time()

        T_s1 = torch.cat([RT_s1, OT_s1], dim=0)
        T_sg = torch.cat([sg]*NR + [sg], dim=0)
        T_sg_targ = torch.cat([sg_targ]*NR + [sg], dim=0)

        T_slst, T_alst = self.simulate(T_s1, T_sg, T_sg_targ, dyn, ktr-1)

        T_slst = T_slst.reshape((T, NR+1, bs, sdim))
        T_alst = T_alst.reshape((T, NR+1, bs, adim))


        RT_slst, OT_slst = torch.split(T_slst, [NR,1], dim=1)
        RT_alst, OT_alst = torch.split(T_alst, [NR,1], dim=1)


        RT_slst = torch.cat([RT_s0.unsqueeze(0).unsqueeze(0).reshape((1,NR,bs,-1)), RT_s1.unsqueeze(0).unsqueeze(0).reshape((1,NR,bs,-1)), RT_slst], dim=0)
        OT_slst = torch.cat([s0.unsqueeze(0).unsqueeze(0), OT_s1.unsqueeze(0).unsqueeze(0), OT_slst], dim=0)


        RT_alst = torch.cat([RT_a0.unsqueeze(0).reshape((1,NR,bs,-1)), RT_alst], dim=0)
        OT_alst = torch.cat([OT_a0.unsqueeze(0).unsqueeze(0), OT_alst], dim=0)


        T_slst = torch.cat([RT_slst, OT_slst], dim=1)
        T_alst = torch.cat([RT_alst, OT_alst], dim=1)
        T_sg = torch.cat([sg]*NR + [sg],dim=0)
        
        T_slst = T_slst.reshape((T+2, bsa, sdim))
        T_alst = T_alst.reshape((T+1, bsa, adim))


        T_score_weighted, T_score = self.score_traj(T_slst, T_alst, T_sg)

        T_score_weighted = T_score_weighted.reshape((T+1, NR+1, bs))
        T_score = T_score.reshape((T+1, NR+1, bs))


        RT_score, OT_score = torch.split(T_score, [NR, 1], dim = 1)
        RT_score_weighted, OT_score_weighted = torch.split(T_score_weighted, [NR, 1], dim = 1)

        R_score = RT_score_weighted.mean(dim=0)
        O_score = OT_score_weighted.mean(dim=0)

        OT_score = OT_score.squeeze(1)

        s_ind = R_score.permute(1,0).argmax(dim=1)

        RT_a0 = RT_a0.reshape((NR,bs,adim)).permute(1,0,2)[torch.arange(bs), s_ind]
        R_score = R_score.max(dim=0).values
        O_score = O_score.flatten()

        O_value = self.pvn.value(s0.detach(), OT_a0.detach(), sg.detach()).flatten()
        R_value = self.pvn.value(s0.detach(), RT_a0.detach(), sg.detach()).flatten()

        l = 0.0
        l += ((O_value - O_score.detach())**2).mean()
        l += ((R_value - R_score.detach())**2).mean()

        if random.uniform(0,1) < 0.001:
            print("RT-score", R_score[0:10])
            print("OT-score", O_score[0:10])
            print("OT-value", O_value[0:10])

        sel_RT = torch.gt(R_score, O_score).float().unsqueeze(1)


        a0_targ = sel_RT * RT_a0 + (1-sel_RT) * OT_a0

        a0_pred = self.pvn.train_act(torch.cat([self.makefeat(s0.detach(),sg.detach()), kemb], dim=1), a0_targ) #Better to use sg (goal) or what we reached (sk_targ).
        
        a0_pred_diverse, kl_loss = self.diverse_pvn.train_act(torch.cat([self.makefeat(s0.detach(),sg.detach()), kemb], dim=1), a0_targ)

        l += self.lfn(a0_pred, a0_targ.detach()).mean()
        l += self.lfn(a0_pred_diverse, a0_targ.detach()).mean()
        l += kl_loss


        return l, a0_pred, torch.cat([s0.unsqueeze(0), OT_slst.squeeze(1)], dim=0).detach(), OT_score


    def loss(self,st,at,stk):
        bs = st.shape[0]

        feat = self.makefeat(st,stk)

        k = torch.Tensor([1]*st.shape[0]).long().cuda()
        kemb = self.koptflag(k)

        a0_pred = self.pvn.mode(torch.cat([feat,kemb],dim=1))

        l = self.lfn(a0_pred, at.detach()).mean()


        return l, a0_pred, [st,stk], None, None

if __name__ == "__main__":


    dyn = torch.load('dyn.pt').cuda() #takes sl, al and returns sn
    score = torch.load('contrastive.pt').cuda()  #takes sl,al,sn and returns score

    dataset = pickle.load(open('data/dataset.p', 'rb'))
    X, A, ast, est = dataset['X'], dataset['A'], dataset['ast'], dataset['est']


    plan = Planner().cuda()

    opt = torch.optim.Adam(plan.parameters(), lr=0.0001)

    plan.score = score
    plan.dyn = dyn

    plan.pvn.decoder_net = torch.jit.script(plan.pvn.decoder_net)
    plan.dyn.dec = torch.jit.script(plan.dyn.dec)
    plan.dyn.enc = torch.jit.script(plan.dyn.enc)

    loss_lst = []
    score_lst = []
    score_approx_lst = []
    splan = None

    for j in range(0,2*300000):

        datak = 32
        st, a = sample_batch(X, A, ast, est, 128, datak) #bs x pos x 2

        ktr = random.choice([1,15])

        if ktr == 1:
            s0 = st[:,0]
            sk = st[:,1]
            a0 = a[:,0]
            l,a0pred, _, plan_score, _ = plan.loss(s0,a0,sk)
        else:
            s0 = st[:,0]
            idx = torch.randperm(s0.shape[0])
            sk = st[:,-1][idx]
            sk_pre = st[:,-2][idx]


            if splan is not None:
                splan_flat = splan.reshape((splan.shape[0]*splan.shape[1], -1))
                splan_flat = splan_flat[torch.randperm(splan_flat.shape[0])][:s0.shape[0]//2]

                s0[s0.shape[0]//2:] = splan_flat
                s0[100:] = st[100:,1]

            a0 = a[:,0]
            l,a0pred,splan, plan_score = plan.multistep(s0,a0,sk,sk_pre,ktr,dyn)


            loss_lst.append(l.item())

        if plan_score is not None:
            bs = plan_score.shape[1]
            score_lst.append(plan_score.max(dim=0).values[:bs//2].mean().item())
            score_approx_lst.append(torch.gt(plan_score.max(dim=0).values[:bs//2], 0.01).float().mean().item())

        l.backward()
        opt.step()
        opt.zero_grad()


        if j % 2000 == 0:
            if len(score_lst)>0:
                print(j, sum(loss_lst)/len(loss_lst))
                print("score-average", sum(score_lst)/len(score_lst))
                print("score-approx-average", sum(score_approx_lst)/len(score_approx_lst))
            score_lst = []
            score_approx_lst = []
            loss_lst = []
            print(ktr)
            print(s0[0],sk[0],a0pred[0])

        if j % 2000 == 0 and ktr != 1:
            viz_plan(s0,sk,splan,plan_score,0,'main')
            worst_ind = plan_score.argmax(dim=0)[:bs//2].argmin()
            viz_plan(s0,sk,splan,plan_score,worst_ind,'worst')

            s0_d = s0[0:1].repeat(128,1)
            sk_d = sk[0:1].repeat(128,1)
             
            roll_s, _ = plan.simulate(s0_d, sk_d, sk_d, plan.dyn, 15, diverse=True)              
            viz_plan_diverse(s0[0:1], sk[0:1], roll_s)

            print("Eval Value")
            vals, s0_grid, sg_grid, sopt, policy_a = plan.eval_value(sk[0:1])
            plt.scatter(s0_grid[:,0], s0_grid[:,1], c=vals, s=120, marker='s', cmap='inferno')
            plt.scatter(x=sk[0][0].item(), y=sk[0][1].item(), color='green')
            plt.savefig('results/vals.png')
            plt.clf()
            print('v min max', vals.min(), vals.max())
            print('sg, opt_s', sk[0:1], sopt)

            policy_a *= 1.0#torch.clamp(policy_a, -0.15, 0.15)
            plt.plot([0.5,0.5], [0.0,0.4], color='black', linewidth=3)
            plt.plot([0.5,0.5], [0.6,1.0], color='black', linewidth=3)
            plt.plot([0.2,0.8], [0.6,0.6], color='black', linewidth=3)
            plt.plot([0.2,0.8], [0.4,0.4], color='black', linewidth=3)
            plt.quiver(s0_grid[:,0], s0_grid[:,1], policy_a[:,0], policy_a[:,1])
            plt.scatter(x=sk[0][0].item(), y=sk[0][1].item(), color='green')
            plt.savefig('results/policy.png')
            plt.clf()









