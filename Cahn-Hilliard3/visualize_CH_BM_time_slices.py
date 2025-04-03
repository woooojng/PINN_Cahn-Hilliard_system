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

time_plotted = 1

###########################################

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

#Choose Test Problem


#Get Test Problem Network
from BM_WZ_CHnet import BM_WZ_CHnet
Net = BM_WZ_CHnet

layers=[3, 128, 128, 128, 128, 128, 128, 1]

net = Net(layers).to(device)

net.load_state_dict(torch.load("CH_Benchmarks_Pass_3.pt", map_location=torch.device('cpu')))

#Graph at various time slices

spatial_discretization = 100

#Define numpy arrays for inputs
x1 = np.linspace(net.x1_l,net.x1_u,spatial_discretization).reshape(spatial_discretization)
x2 = np.linspace(net.x2_l,net.x2_u,spatial_discretization).reshape(spatial_discretization)
x1x2 = np.array(np.meshgrid(x1, x2)).reshape(2,spatial_discretization**2)



x1_input = x1x2[0].reshape(spatial_discretization**2, 1)
x2_input = x1x2[1].reshape(spatial_discretization**2, 1)

x1x2 = [x1_input, x2_input]

#convert to pytorch tensors
pt_x1 = Variable(torch.from_numpy(x1_input).float(), requires_grad=False).to(device)
pt_x2 = Variable(torch.from_numpy(x2_input).float(), requires_grad=False).to(device)

def plot_slice(time_plotted):
    
    t = time_plotted*np.ones((spatial_discretization**2,1))
    pt_t = Variable(torch.from_numpy(t).float(), requires_grad=False).to(device)
    
    #get network outputs
    pt_phi, pt_mu = net(pt_x1, pt_x2, pt_t)
    
    #Convert back to numpy
    phi = pt_phi.detach().numpy()

    return phi

X, Y = np.meshgrid(x1, x2)


time_slices = [0, .1, .25, .5, .75, 1]

fig, axs = plt.subplots(3,2)
fig.tight_layout()

for i in range(len(time_slices)):
    axs[i//2, i%2].set_title(f'Time={time_slices[i]}')
    im = axs[i//2, i%2].pcolor(X, Y, plot_slice(time_slices[i]).reshape(X.shape), vmin=-1, vmax=1)


fig.colorbar(im, ax=axs, orientation='vertical', fraction=0.02, pad=0.02)


plt.show()
