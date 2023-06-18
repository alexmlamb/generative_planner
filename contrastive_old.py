
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
import random

def make_prod(S,A,SN):
    row_indices = torch.arange(S.shape[0]).cuda()
    combinations = torch.cartesian_prod(row_indices, row_indices)
    st = S[combinations[:, 0]]
    sn = SN[combinations[:, 1]]
    a = A[combinations[:,0]]

    return st, a, sn

class GaussianFourierProjectionTime(nn.Module):
  """Gaussian Fourier embeddings for noise levels."""

  def __init__(self, embedding_size=256, scale=1.0):
    super().__init__()
    self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

  def forward(self, x):
    x_shape = x.shape
    x = x.flatten()
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
    out = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

    out = out.reshape((x_shape[0], 256*4))

    return out


class Contrastive(nn.Module):
    def __init__(self):
        super(Contrastive, self).__init__()

        self.gauss_feat = GaussianFourierProjectionTime().cuda()

        self.kemb = nn.Embedding(5,256)

        self.net1 = nn.Sequential(nn.Linear(1024+1024+256, 1024), nn.Dropout(0.2), nn.LeakyReLU(), nn.Linear(1024,512), nn.LeakyReLU(), nn.Linear(512,512), nn.LeakyReLU(), nn.Linear(512, 256))
        self.net2 = nn.Sequential(nn.Linear(1024+256, 1024), nn.Dropout(0.2), nn.LeakyReLU(), nn.Linear(1024,512), nn.LeakyReLU(), nn.Linear(512,512), nn.LeakyReLU(), nn.Linear(512, 256))

    #Shaped (Tx2), (Tx2), (Tx2).  Classify the s_pos and a_pos with next-vals as positive, classify s_pos with s_neg as negatives.  
    def forward(self, s, a, s_n):


        score_logits = 5.0 - torch.sqrt(((s - s_n)**2).sum(dim=1))

        #inp = torch.cat([s,a*0.0,s_n],dim=1)

        #score_logits = self.net(inp)
        score = F.sigmoid(score_logits)

        return score_logits, score #loss is scalar, score is (T-vector).  

    def forward_enc(self, s, a, s_n, k):

        kfeat = self.kemb(torch.Tensor([k]*s.shape[0]).long().cuda())

        s = self.net1(torch.cat([self.gauss_feat(s), self.gauss_feat(a), kfeat],dim=1))
        s_n = self.net2(torch.cat([self.gauss_feat(s_n),kfeat],dim=1))

        return self.forward(s, a, s_n)

    def pos_contrastive_loss(self, s, a, spos, k):
        s = s.reshape((s.shape[0]*s.shape[1], -1))
        a = a.reshape((a.shape[0]*a.shape[1], -1))
        spos = spos.reshape((spos.shape[0]*spos.shape[1], -1))

        kfeat = self.kemb(torch.Tensor([k]*s.shape[0]).long().cuda())

        s = self.net1(torch.cat([self.gauss_feat(s), self.gauss_feat(a), kfeat], dim=1))
        spos = self.net2(torch.cat([self.gauss_feat(spos),kfeat],dim=1))


        return self.forward(s,a,spos)

    def neg_contrastive_loss(self, s, a, sneg, k):

        #get all combinations of s,a,sneg

        s = s.reshape((s.shape[0]*s.shape[1], -1))
        a = a.reshape((a.shape[0]*a.shape[1], -1))
        sneg = sneg.reshape((sneg.shape[0]*sneg.shape[1], -1))

        kfeat = self.kemb(torch.Tensor([k]*s.shape[0]).long().cuda())

        s = self.net1(torch.cat([self.gauss_feat(s), self.gauss_feat(a), kfeat],dim=1))
        sneg = self.net2(torch.cat([self.gauss_feat(sneg),kfeat],dim=1))

        s_c, a_c, sn_c = make_prod(s, a, sneg)

        return self.forward(s_c,a_c,sn_c)

    #Call contrastive and classify.  
    def loss(self, s_seq, a_seq, s_neg):

        assert s_seq.shape[0] == s_neg.shape[0]

        k = random.randint(1,2)

        s_last = s_seq[:-k]
        a_last = a_seq[:-k]
        s_pos = s_seq[k:]
        s_neg = s_neg[k:] #not needed, just to make sizes match   

        s_last_neg = s_last * 1.0

        if False and random.uniform(0,1) < 0.5:
            noise_scale = random.uniform(0.01,0.2)
            s_neg += (torch.randn_like(s_neg) * noise_scale).round(decimals=2)
            s_last_neg += (torch.randn_like(s_last_neg) * noise_scale).round(decimals=2)

        pos_logits, _ = self.pos_contrastive_loss(s_last, a_last, s_pos, k)
        neg_logits, _ = self.neg_contrastive_loss(s_last_neg, a_last, s_neg, k)

        l = 0.0
        bce = nn.BCEWithLogitsLoss()
        l += bce(pos_logits, torch.ones_like(pos_logits))
        l += bce(neg_logits, torch.zeros_like(neg_logits))
        return l



