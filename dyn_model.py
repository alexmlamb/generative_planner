
import pickle
from utils import sample_batch
import torch
import torch.nn as nn
import numpy as np

'''
Idea: 
    -Get tuples of s[t], a[t], s[t+1] and train AE with gaussian NLL.  Off-manifold NLL (wall-crossing) should be very bad, compared to normal transitions.  Do we need VAE on bottleneck?  Probably yes in concept to prevent memorizing.  

    -Study how NLL changes around the discrete peaks.  

'''

class GaussianFourierProjectionTime(nn.Module):
  """Gaussian Fourier embeddings for noise levels."""

  def __init__(self, embedding_size=256, scale=1.0):
    super().__init__()
    self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class DynVAE(nn.Module):

    def __init__(self):
        super().__init__()

        #self.trans = nn.Sequential(nn.Linear(2+2, 1024), nn.Dropout(0.5), nn.LeakyReLU(), nn.Linear(1024,1024), nn.Dropout(0.5), nn.LeakyReLU(), nn.Linear(1024,1024), nn.Dropout(0.5), nn.LeakyReLU(), nn.Linear(1024,4))

        self.gauss_feat = GaussianFourierProjectionTime().cuda()

        self.enc = nn.Sequential(nn.Linear(512*4, 1024), nn.Dropout(0.0), nn.LeakyReLU(), nn.Linear(1024,1024), nn.Dropout(0.0), nn.LeakyReLU(), nn.Linear(1024, 64*2), nn.Dropout(0.0))
        self.dec = nn.Sequential(nn.Linear(64+512*4, 1024), nn.Dropout(0.0), nn.LeakyReLU(), nn.Linear(1024,1024), nn.Dropout(0.0), nn.LeakyReLU(), nn.Linear(1024, 6*2))

    def forward(self,sl,al,sn=None):

        bs = sl.shape[0]

        #print(sa.shape) #bs x 4 x 32

        #sa_l = sa[:,:,:-1]
        #sa_n = sa[:,:,1:]
        #s_n = sa_n[:,:2,:]

        inp = torch.cat([sl,al],dim=1)

        inp_flat = inp.reshape((inp.shape[0]*inp.shape[1]))
        gauss = self.gauss_feat(inp_flat)
        gauss = gauss.reshape((bs,-1))

        if sn is not None:
            targ = torch.cat([sl,al,sn],dim=1)

        h = self.enc(gauss)

        h_mu = h[:,:64]
        h_std = torch.exp(h[:,64:])

        mu_prior = torch.zeros_like(h_mu)
        std_prior = torch.ones_like(h_std)
        prior = torch.distributions.normal.Normal(mu_prior, std_prior)
        posterior = torch.distributions.normal.Normal(h_mu, h_std)
        kl_loss = torch.distributions.kl_divergence(posterior, prior).sum(dim=-1).mean()

        h = posterior.rsample()

        h = torch.cat([h,gauss],dim=1)

        out = self.dec(h)

        #loss = torch.abs(out - inp.detach()).mean()

        mu = out[:,:6]
        std = torch.exp(out[:,6:])

        if sn is not None:
            log_loss = torch.log(std * 2.5066) + 0.5 * (mu - targ.detach())**2 / std**2
            return log_loss.mean() + kl_loss * 0.01, log_loss.mean().item(), mu.round(decimals=3)
        else:
            return mu[:,-2:]

def train_dynamics(S,A):

    net = DynVAE().cuda()
    opt = torch.optim.Adam(net.parameters(), lr = 0.0001)

    for j in range(0, 200000): 
        k = 3
        st, a = sample_batch(None, A, S, None, 128, k)

        sl = st[:,0]
        sn = st[:,1]
        al = a[:,0]

        loss = net(sl, al, sn)[0]

        if j % 500 == 0:
            print(j, loss)

        loss.backward()
        opt.step()
        opt.zero_grad()
        
        if j % 2000 == 0:
            print("Test cases")
            print("Low")
            print(net(torch.Tensor([[0.4,0.5]]).cuda(), torch.Tensor([[0.1,0.0]]).cuda(), torch.Tensor([[0.5,0.5]]).cuda()))
            print("Low")
            print(net(torch.Tensor([[0.1,0.1]]).cuda(), torch.Tensor([[0.1,0.1]]).cuda(), torch.Tensor([[0.2,0.2]]).cuda()))
            print("High")
            print(net(torch.Tensor([[0.4,0.3]]).cuda(), torch.Tensor([[0.1,0.0]]).cuda(), torch.Tensor([[0.5,0.3]]).cuda()))
            print("High")
            print(net(torch.Tensor([[0.45,0.2]]).cuda(), torch.Tensor([[0.1,0.0]]).cuda(), torch.Tensor([[0.55,0.2]]).cuda()))

        if j % 5000 == 0:
            torch.save(net, 'dyn.pt')

    return net


if __name__ == "__main__":


    dataset = pickle.load(open('data/dataset.p', 'rb'))
    X, A, ast, est = dataset['X'], dataset['A'], dataset['ast'], dataset['est']


    train_dynamics(ast, A)



