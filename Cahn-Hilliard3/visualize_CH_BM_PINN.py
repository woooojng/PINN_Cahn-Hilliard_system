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


#Get Test Problem Network
from BM_WZ_CHnet import BM_WZ_CHnet
Net = BM_WZ_CHnet

layers=[3, 128, 128, 128, 128, 128, 128, 1]

net = Net(layers).to(device)

net.load_state_dict(torch.load("WZ_Regularized_BFGS.pt", map_location=torch.device('cpu')))

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
pt_phi, pt_mu = net(pt_x1, pt_x2, pt_t)
pt_Psi = net.W(pt_x1, pt_x2, pt_t)

#get actual initial condition
IC_exact = Net.Initial_Condition_phi(net, pt_x1, pt_x2)

#Convert back to numpy
phi = pt_phi.detach().numpy()
IC_exact = IC_exact.detach().numpy()
Psi = pt_Psi.detach().numpy()
IC_error = phi-IC_exact

X, Y = np.meshgrid(x1, x2)

fig, axs = plt.subplots(3)

#fig.suptitle(f'Time = {time_plotted}')
fig.tight_layout()
axs[0].set_title('Predicted Density')
axs[1].set_title('Actual Initial Density')
axs[2].set_title('Error')
im = axs[0].pcolor(X, Y, phi.reshape(X.shape), vmin=-1, vmax=1)
im2 = axs[1].pcolor(X, Y, IC_exact.reshape(X.shape), vmin=-1, vmax=1)
im3 = axs[2].pcolor(X, Y, IC_error.reshape(X.shape), vmin=-1, vmax=1)

fig.colorbar(im, ax=axs, orientation='vertical', fraction=0.02, pad=0.02)


plt.show()
