
import numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

def viz_plan(s0, sk, splan, plan_score, index, title):

    planlst = list(map(lambda sp: list(map(lambda z: round(z, 2), sp[index].cpu().data.numpy().tolist())), splan))
    print("Plan", list(map(lambda sp: list(map(lambda z: round(z, 2), sp[index].cpu().data.numpy().tolist())), splan)))
    print("Plan score", plan_score.max(dim=0).values[index])

    plt.plot([0.5,0.5], [0.0,0.4], color='black', linewidth=3)
    plt.plot([0.5,0.5], [0.6,1.0], color='black', linewidth=3)
    plt.plot([0.2,0.8], [0.6,0.6], color='black', linewidth=3)
    plt.plot([0.2,0.8], [0.4,0.4], color='black', linewidth=3)

    plt.plot(numpy.array(planlst)[:,0], numpy.array(planlst)[:,1], linewidth=3, **{'color': 'lightsteelblue', 'marker': 'o'}, zorder=1)

    plt.scatter(x=s0[index][0].item(), y=s0[index][1].item(), color='blue',zorder=2)
    plt.scatter(x=sk[index][0].item(), y=sk[index][1].item(), color='green',zorder=3)

    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.0])

    plt.savefig('results/plan_%s.png' % title)
    plt.clf()



def viz_plan_diverse(s0, sk, splan):

    plt.plot([0.5,0.5], [0.0,0.4], color='black', linewidth=3)
    plt.plot([0.5,0.5], [0.6,1.0], color='black', linewidth=3)
    plt.plot([0.2,0.8], [0.6,0.6], color='black', linewidth=3)
    plt.plot([0.2,0.8], [0.4,0.4], color='black', linewidth=3)

    splan = torch.cat([s0.unsqueeze(0).repeat(1,splan.shape[1],1), splan], dim=0)

    print('shapes', s0.shape, sk.shape, splan.shape)

    for index in range(splan.shape[1]):
        planlst = list(map(lambda sp: list(map(lambda z: round(z, 2), sp.cpu().data.numpy().tolist())), splan[:,index]))
        plt.plot(numpy.array(planlst)[:,0], numpy.array(planlst)[:,1], linewidth=1, **{'marker': 'o'}, zorder=1, alpha=0.5)

    plt.scatter(x=s0[0,0].item(), y=s0[0,1].item(), color='blue',zorder=2)
    plt.scatter(x=sk[0,0].item(), y=sk[0,1].item(), color='green',zorder=3)

    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.0])

    plt.savefig('results/plan_diverse.png')
    plt.clf()







