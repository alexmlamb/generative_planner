
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np

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

        self.net1 = nn.Sequential(nn.Linear(4*(2+2), 1024), nn.LeakyReLU(), nn.Linear(1024,512), nn.LeakyReLU(), nn.Linear(512,512), nn.LeakyReLU(), nn.Linear(512, 1))

    #Shaped (Tx2), (Tx2), (Tx2).  Classify the s_pos and a_pos with next-vals as positive, classify s_pos with s_neg as negatives.  
    def forward(self, s, a, s_n):


        score_logits = 5.0 - torch.sqrt(((s - s_n)**2).sum(dim=1))

        #inp = torch.cat([s,a*0.0,s_n],dim=1)

        #score_logits = self.net(inp)
        score = F.sigmoid(score_logits)

        return score_logits, score #loss is scalar, score is (T-vector).  

    def forward_enc(self, s, a):
 
        s = s.unsqueeze(0)
        a = a.unsqueeze(0)
       
        out = torch.cat([s,a],dim=2)

        out = out.reshape((out.shape[0], -1))

        out = self.net1(out)

        return F.sigmoid(out)

        #s = self.net1(torch.cat([self.gauss_feat(s), self.gauss_feat(a)],dim=1))
        #s_n = self.net2(self.gauss_feat(s_n))


    def pos_contrastive_loss(self, s, a, spos):
        s = s.reshape((s.shape[0]*s.shape[1], -1))
        a = a.reshape((a.shape[0]*a.shape[1], -1))
        spos = spos.reshape((spos.shape[0]*spos.shape[1], -1))


        s = self.net1(torch.cat([self.gauss_feat(s), self.gauss_feat(a)], dim=1))
        spos = self.net2(self.gauss_feat(spos))

        return self.forward(s,a,spos)

    def neg_contrastive_loss(self, s, a, sneg):

        #get all combinations of s,a,sneg

        s = s.reshape((s.shape[0]*s.shape[1], -1))
        a = a.reshape((a.shape[0]*a.shape[1], -1))
        sneg = sneg.reshape((sneg.shape[0]*sneg.shape[1], -1))

        s = self.net1(torch.cat([self.gauss_feat(s), self.gauss_feat(a)],dim=1))
        sneg = self.net2(self.gauss_feat(sneg))

        s_c, a_c, sn_c = make_prod(s, a, sneg)

        return self.forward(s_c,a_c,sn_c)

    #Call contrastive and classify.  
    def loss(self, s_seq, a_seq, s_neg):


        idx = torch.randperm(s_seq.shape[1])

        assert s_seq.shape[0] == s_neg.shape[0]

        seq = torch.cat([s_seq, a_seq], dim=2)
        kcut = 2

        seg_left = seq[0:kcut]
        seg_right = seq[kcut:].permute(1,0,2)[idx].permute(1,0,2)
 
        neg = torch.cat([seg_left, seg_right], dim=0)


        seq = seq.permute(1,0,2)
        neg = neg.permute(1,0,2)


        seq = seq.reshape((seq.shape[0], -1))
        neg = neg.reshape((neg.shape[0], -1))


        pos_logits = self.net1(seq)
        neg_logits = self.net1(neg)


        #pos_logits, _ = self.pos_contrastive_loss(s_last, a_last, s_pos)
        #neg_logits, _ = self.neg_contrastive_loss(s_last, a_last, s_neg)

        l = 0.0
        bce = nn.BCEWithLogitsLoss()
        l += bce(pos_logits, torch.ones_like(pos_logits))
        l += bce(neg_logits, torch.zeros_like(neg_logits))
        return l







