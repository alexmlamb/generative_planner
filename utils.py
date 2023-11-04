
import random
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sample_example(X, A, ast, est, k):
    N = A.shape[0]
    t = random.randint(0, N - 200)
    #k = random.randint(1, max_k)

    ret_seq = True
    ret_state = True

    if ret_seq:
        return (ast[t:t+k], A[t:t+k])
    elif ret_state:
        return (ast[t], ast[t + 1], ast[t + k], k, A[t], ast[t], est[t])
    else:
        return (X[t], X[t + 1], X[t + k], k, A[t], ast[t], est[t])


def sample_batch(X, A, ast, est, bs, k):
    xt = []
    xtn = []
    xtk = []
    klst = []
    astate = []
    estate = []
    alst = []

    for b in range(bs):
        lst = sample_example(X, A, ast, est, k=k)
        xt.append(lst[0])
        #xtn.append(lst[1])
        #xtk.append(lst[2])
        #klst.append(lst[3])
        alst.append(lst[1])
        #astate.append(lst[5])
        #estate.append(lst[6])

    xt = torch.Tensor(np.array(xt)).to(device)
#    xtn = torch.Tensor(np.array(xtn)).to(device)
#    xtk = torch.Tensor(np.array(xtk)).to(device)
#    klst = torch.Tensor(np.array(klst)).long().to(device)
    alst = torch.Tensor(np.array(alst)).to(device)
#    astate = torch.Tensor(np.array(astate)).to(device)
#    estate = torch.Tensor(np.array(estate)).to(device)

    return xt, alst#xtn, xtk, klst, alst, astate, estate


# --------------------------------------------------


from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import random
import pickle

class CustomDataset(Dataset):
    """Face Landmarks dataset."""
    def __init__(self):

        self.dataset = pickle.load(open('data/dataset.p', 'rb'))
        self.slen = 3
        self.n = self.dataset['A'].shape[0]

    def __len__(self):
        return self.n-self.slen-2

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        r_ind = random.randint(0, self.n-self.slen-2)

        s_seq = torch.Tensor(self.dataset['ast'][idx:idx+self.slen])
        a_seq = torch.Tensor(self.dataset['A'][idx:idx+self.slen])
        s_neg = torch.Tensor(self.dataset['ast'][r_ind:r_ind+self.slen])

        return s_seq, a_seq, s_neg

if __name__ == "__main__":
    data = CustomDataset()
    loader = DataLoader(data, batch_size=int(256), num_workers=2)
    for i, (sample_batched) in enumerate(loader):
        x, y, z = sample_batched
        print('done')