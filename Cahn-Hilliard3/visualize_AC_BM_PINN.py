# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 13:52:04 2023

@author: kevbuck
"""

import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt

###########################################

time_plotted = 0

###########################################

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

#Choose Test Problem
test = 1

#Get Test Problem Network
if test == 1:
    from BM_I_ACnet import BM_I_ACnet
    Net = BM_I_ACnet
elif test == 2:
    from BM_II_CHnet import BM_II_CHnet
    Net = BM_II_CHnet
elif test == 3:
    from BM_III_CHnet import BM_III_CHnet
    Net = BM_III_CHnet
elif test == 4:
    from BM_IV_CHnet import BM_IV_CHnet
    Net = BM_IV_CHnet

layers=[3,40, 40, 40, 40,1]

net = Net(layers).to(device)

net.load_state_dict(torch.load("CH_Benchmarks_Pass_3.pt", map_location=torch.device('cpu')))

time_plotted = time_plotted/net.acceleration_factor

#Graph at various time slices

spatial_discretization = 100

#Define numpy arrays for inputs
x1 = np.linspace(net.x1_l,net.x1_u,spatial_discretization).reshape(spatial_discretization)
x2 = np.linspace(net.x2_l,net.x2_u,spatial_discretization).reshape(spatial_discretization)
x1x2 = np.array(np.meshgrid(x1, x2)).reshape(2,spatial_discretization**2)

t = time_plotted*np.ones((spatial_discretization**2,1))

x1_input = x1x2[0].reshape(spatial_discretization**2, 1)
x2_input = x1x2[1].reshape(spatial_discretization**2, 1)

x1x2 = [x1_input, x2_input]

#convert to pytorch tensors
pt_x1 = Variable(torch.from_numpy(x1_input).float(), requires_grad=False).to(device)
pt_x2 = Variable(torch.from_numpy(x2_input).float(), requires_grad=False).to(device)
pt_t = Variable(torch.from_numpy(t).float(), requires_grad=False).to(device)

#get network outputs
pt_phi = net(pt_x1, pt_x2, pt_t)

#get actual initial condition
IC_exact = Net.Initial_Condition(net, pt_x1, pt_x2)

#Convert back to numpy
phi = pt_phi.data.cpu().numpy()
IC_exact = IC_exact.data.cpu().numpy()
IC_error = phi-IC_exact

X, Y = np.meshgrid(x1, x2)

fig, axs = plt.subplots(3)
#fig.suptitle(f'Time = {time_plotted}')
fig.tight_layout()
axs[0].set_title('Predicted Initial Density')
axs[1].set_title('Actual Initial Density')
axs[2].set_title('Error')
axs[0].pcolor(X, Y, phi.reshape(X.shape))
axs[1].pcolor(X, Y, IC_exact.reshape(X.shape))
axs[2].pcolor(X, Y, IC_error.reshape(X.shape))

#plt.colorbar(ax=ax[0])

#plt.colorbar()
plt.show()
