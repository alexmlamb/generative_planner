'''
Algorithm: 

    -Files each with train method and a class: 
        -contrastive.  Can test this and verify that the wall is understood well (crossing the wall should have score of 0).  
        -AC
        -generator

    -Step 0: Get the hallway environment setup and load the dataset.  Write sub-seq loader and random sample loader.  
    -Step 1: Sample k=20 steps from the buffer, and use to train Psi(s,a,s'), p_g(s[1:k], a[1:k] | s[1], s[k], a[k], opt=False), p_a(a[1] | s[1], s[k], opt=False).  Should work on 1-step or few steps.  Verify walls.  
    -Step 2: Write scoring code for hypothetical (s,a) sequences and get dyn_score and goal_score.  
    -Step 3: Pick pairs of (s,s_g) uniformly, then learn optimal versions of p_g and p_a based on hitting goal with decent dyn_score.  

-Get s sequence of size (T, 2) with a (T,2).  
-Maybe T=32 is reasonable.  32/16/8/4 1d convolutions.  

Psi: (256,2,256) -> 2 MLP.  Run in parallel over time-steps by taking s[1:] and s[:-1], a[:-1].  Also get a random negative batch s_r of totally random points.    
p_g: 1D-Conv-VAE, with 256+2 channels per-position.  Enc/Dec/Prior.  dec/prior condition on is_opt.  
p_a: (256,256,k_embed,is_opt) --> 2 MLP.  

---

  We can also give a (k,s_gk) sub-goal, then only give it goal-score for first k-steps, and take first action to reach the sub-goal.  (So then we get a generative model which also knows how to reach two different goals in its allotted time).  On the other hand, just training with empirical rollouts with a good policy may also accomplish this asymptotically since we'll train on trajectories between two different goals.  

---
Final algorithm:
    -Learn representations from data as we keep planning and updating representation.  
    -Keeping sampling from empirical buffer while doing steps 1-3 in tandem.  
    -When new data is collected, pick s_g uniformly, then take one random action then k-1 from policy network.  Can also stop if Psi(s,a,s_g) > 0.9, and then try to get to a new goal.  

'''

import random
import torch
import torch.nn as nn
import pickle

ctype = 'v1'
if ctype == 'v1':
    from contrastive_old import Contrastive
elif ctype == 'v2':
    from contrastive import Contrastive
import time

score_net = Contrastive().cuda()

opt = torch.optim.Adam(list(score_net.parameters()), lr=0.0001)

if __name__ == "__main__":

    import time
    t0 = time.time()

    dataset = pickle.load(open('data/dataset.p', 'rb'))
    n = dataset['X'].shape[0]
    slen = 3
    bs = 256
    print('Num samples', dataset['X'].shape[0])       
    assert len(dataset['X'].shape) == 4
    assert len(dataset['A'].shape) == 2
    action_dim = dataset['A'].shape[1]

    S = dataset['ast']
    A = dataset['A']

    print(dataset['ast'].shape)
    print(dataset['A'].shape)
 
    contrastive_loss = []

    for j in range(0,400000):

        s_seq, a_seq, s_neg = [], [], []
        for i in range(bs):
            k_ind = random.randint(0,n-slen-2)
            r_ind = random.randint(0,n-slen-2)
            s_seq.append(torch.Tensor(S[k_ind : k_ind + slen]).cuda().unsqueeze(1))
            a_seq.append(torch.Tensor(A[k_ind : k_ind + slen]).cuda().unsqueeze(1))
            s_neg.append(torch.Tensor(S[r_ind : r_ind + slen]).cuda().unsqueeze(1))

        s_seq = torch.cat(s_seq,dim=1)
        a_seq = torch.cat(a_seq,dim=1)
        s_neg = torch.cat(s_neg,dim=1)

        loss = 0.0
    
        closs = score_net.loss(s_seq, a_seq, s_neg)
        contrastive_loss.append(closs.item())

        loss += closs

        opt.zero_grad()
        loss.backward()
        opt.step()

        if j % 500 == 0:
            print('time', time.time() - t0)
            t0 = time.time()
            print('loss', j, sum(contrastive_loss)/len(contrastive_loss))

            contrastive_loss = []


        if ctype=='v2' and j % 500 == 0:
            test_neg1 = torch.Tensor([[0.45, 0.2, 0.2, 0.0], [0.65, 0.2, 0.07, 0.0], [0.72, 0.2, 0.05, 0.0], [0.77, 0.2, 0.05, 0.0]]).cuda()
            test_neg2 = torch.Tensor([[0.45, 0.2, 0.2, 0.0], [0.65, 0.2, 0.0, 0.0], [0.65, 0.2, 0.05, 0.0], [0.7, 0.2, 0.0, 0.0]]).cuda()

            test_pos1 = torch.Tensor([[0.15, 0.2, 0.1, 0.0], [0.25, 0.2, 0.0, 0.0], [0.25, 0.2, 0.00, 0.0], [0.25, 0.2, 0.0, 0.0]]).cuda()
            test_pos2 = torch.Tensor([[0.55, 0.2, 0.1, 0.0], [0.65, 0.2, 0.0, 0.0], [0.65, 0.2, 0.00, 0.0], [0.65, 0.2, 0.0, 0.0]]).cuda()

            print('neg-cross1', score_net.forward_enc(test_neg1[:,0:2],test_neg1[:,2:4]))
            print('neg-cross2', score_net.forward_enc(test_neg2[:,0:2],test_neg2[:,2:4]))

            print('pos-cross1', score_net.forward_enc(test_pos1[:,0:2],test_pos1[:,2:4]))
            print('pos-cross2', score_net.forward_enc(test_pos2[:,0:2],test_pos2[:,2:4]))
            #print('pos-cross', score_net.forward_enc(test_pos[:,0:2], test_pos[:,2:4], test_pos[:,4:6])[1])
            #print('err-cross', score_net.forward_enc(test_err[:,0:2], test_err[:,2:4], test_err[:,4:6])[1])

        if ctype=='v1' and j % 500 == 0:
            test_neg = torch.Tensor([[0.45, 0.2, 0.2, 0.0, 0.65, 0.2], [0.45, 0.2, 0.07, 0.0, 0.52, 0.2], [0.45, 0.2, 0.06, 0.0, 0.51, 0.2], [0.45, 0.2, 0.1, 0.0, 0.55, 0.2], [0.45, 0.2, 0.15, 0.0, 0.52, 0.2]]).cuda()

            test_pos = torch.Tensor([[0.45, 0.2, 0.2, 0.0, 0.49, 0.2], [0.48, 0.1, 0.1, 0.0, 0.49, 0.1], [0.1, 0.1, 0.1, 0.0, 0.4, 0.1]]).cuda()

            test_err = torch.Tensor([[0.2, 0.2, 0.0, 0.0, 0.22, 0.2], [0.5, 0.5, 0.2, 0.0, 0.71, 0.5]]).cuda()

            print('neg-cross', score_net.forward_enc(test_neg[:,0:2], test_neg[:,2:4], test_neg[:,4:6],1)[1])
            print('pos-cross', score_net.forward_enc(test_pos[:,0:2], test_pos[:,2:4], test_pos[:,4:6],1)[1])
            print('err-cross', score_net.forward_enc(test_err[:,0:2], test_err[:,2:4], test_err[:,4:6],1)[1])

            torch.save(score_net, 'contrastive.pt')


