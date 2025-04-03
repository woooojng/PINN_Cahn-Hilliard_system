# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 13:11:16 2025

@author: kevbuck
"""

import matplotlib.pyplot as plt
import torch
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

from BM_WZ_CHnet import BM_WZ_CHnet
Net = BM_WZ_CHnet

layers=[3, 128, 128, 128, 128, 128, 128, 1]
net = Net(layers).to(device)

x1_l = net.x1_l
x1_u = net.x1_u
x2_l = net.x2_l
x2_u = net.x2_u

i=15000

x_IC = torch.load(f'x_IC_adaptive_{i}.pt', map_location=torch.device('cpu'))
y_IC = torch.load(f'y_IC_adaptive_{i}.pt', map_location=torch.device('cpu'))


x_dom = torch.load(f'x_dom_adaptive_{i}.pt', map_location=torch.device('cpu'))
y_dom = torch.load(f'y_dom_adaptive_{i}.pt', map_location=torch.device('cpu'))
t_dom = torch.load(f't_dom_adaptive_{i}.pt', map_location=torch.device('cpu'))

try:
    x_IC = x_IC.detach()
    y_IC = y_IC.detach()
    x_dom = x_dom.detach()
    y_dom = y_dom.detach()
    t_dom = t_dom.detach()
except:
    pass

fig, axs = plt.subplots(2)
fig.suptitle(f'Epoch = {i}')
fig.tight_layout()

axs[0].set_title('IC Adaptive Points')
axs[1].set_title('Domain Adaptive Points')


axs[0].scatter(x_IC, y_IC, s=1)
axs[0].set_xlim(x1_l, x1_u)
axs[0].set_ylim(x2_l, x2_u)
axs[1].scatter(x_dom, y_dom, c=t_dom.detach().numpy(), s=1)
axs[1].set_xlim(x1_l, x1_u)
axs[1].set_ylim(x2_l, x2_u)
#axs[1].colorbar()

plt.show()



