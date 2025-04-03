#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import numpy as np
from torch.autograd import Variable

import time
import shutil
import os

from datetime import datetime

from Building_Net import Net
from Conditions import lossIC, lossBdry,lossNSpde

currentDateTime = datetime.now()
print("Date of Today : ", currentDateTime.month, " /", currentDateTime.day, "\nHour : ", currentDateTime.hour) 
ctime = f"{currentDateTime.month}_{currentDateTime.day}_{currentDateTime.hour}h"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




def create_network(load, loadfile):
    
    net = Net()
    net = net.to(device)
    epsilon = []
    #Attempt to load the saved pt. file
    if load == True:
        try:
            net.load_state_dict(torch.load(loadfile, map_location=torch.device(device)))
        except:
            print("\nLoading file was failed\n")
        else:
            print("\nLoading file was completed\n")
    
    print('Training PDE')
    start = time.time() #initialize tracking computational time
    
    partial_time_set = [0, 0, 0] #initialize time recording list with number of learnning rates on whole domain training
    
    for i in range(len(partial_time_set)):
        if i == 0:
            #First loop uses progressively increasing time intervals
            print(f'\n\nTraining Pass {i+1}')
            time_slices = np.array([1])
            iterations = 30000 #iterations for each learning rate
            learning_rate = 10**-3
        elif i == 1:
            print(f'\n\nTraining Pass {i+1}')
            #time_slices = [time_slices[-1]]
            time_slices = [time_slices[-1]]
            iterations = 0 #iterations for each learning rate
            learning_rate = 10**-4
        elif i == 2:
            print(f'\n\nTraining Pass {i+1}')
            time_slices = [time_slices[-1]]
            iterations = 0 #iterations for each learning rate
            learning_rate = 10**-4
        
        training_loop(net, time_slices, iterations, learning_rate, record_loss = 100, print_loss = 500, epsilon = epsilon)
        torch.save(net.state_dict(), f"{ctime}_Training_{i+1}.pt")
        partial_time_set[i] = time.time()
        np.savetxt(f"{ctime}epsilon_{i}.txt", epsilon)

    print("Total Time:\t", partial_time_set[-1]-start, '\nPass 1 Time:\t', partial_time_set[0]-start, 
          '\nPass 2 Time:\t', partial_time_set[1]-partial_time_set[0], '\nPass 3 Time:\t', partial_time_set[2]-partial_time_set[1])
   
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def training_loop(net, time_slices, iterations, learning_rate, record_loss, print_loss, epsilon):
    
    # Domain boundary values
    x_l = net.x1_l
    x_u = net.x1_u
    
    #time starts at 0, ends at upper bouund updated in time_slices
    t_l = 0

    #numbers of sampling collocation points on each part
    IC_collocation = int(150)
    BC_collocation = int(200)
    pde_collocation = int(5000)

    #update the learning rate as defined
    for g in net.optimizer.param_groups:
        g['lr'] = learning_rate
    
    #Iterate over time slices
    for final_time in time_slices:
        with torch.autograd.no_grad():
            print("\n\nCurrent End Time:", final_time, "Current Learning Rate: ", get_lr(net.optimizer))  
        epoch = 0
        for epoch in range(1, iterations):

            # initialize gradients to zero
            net.optimizer.zero_grad()

            ##Define input points
            x_IC = np.random.uniform(low=x_l, high=x_u, size=(IC_collocation,1))
            t_IC = np.random.uniform(low=t_l, high=t_l, size=(IC_collocation,1))
            
            input_x_IC = Variable(torch.from_numpy(x_IC).float(), requires_grad=True).to(device)
            input_t_IC = Variable(torch.from_numpy(t_IC).float(), requires_grad=True).to(device)
            
            x_BC = np.random.uniform(low=x_l, high=x_u, size=(BC_collocation,1))
            t_BC = np.random.uniform(low=t_l, high=final_time, size=(BC_collocation,1))
    
            input_x_BC= Variable(torch.from_numpy(x_BC).float(), requires_grad=True).to(device)
            input_t_BC = Variable(torch.from_numpy(t_BC).float(), requires_grad=True).to(device)
    
            x_domain = np.random.uniform(low= x_l, high=x_u, size=(pde_collocation, 1))
            t_domain = np.random.uniform(low= t_l, high=final_time, size=(pde_collocation, 1)) 

            input_x_domain = Variable(torch.from_numpy(x_domain).float(), requires_grad=True).to(device)
            input_t_domain = Variable(torch.from_numpy(t_domain).float(), requires_grad=True).to(device)

            #Take additive appaptive sampling with 500 highest loss points
            PDEloss_list = []
            for i in range(pde_collocation):
                if i ==0:
                    PDEloss_tensor = lossNSpde(net, input_x_domain[i,0].reshape((1,1)), input_t_domain[i,0].reshape((1,1))).reshape((1,1))
                else:
                    PDEloss_tensor = torch.cat((PDEloss_tensor, lossNSpde(net, input_x_domain[i,0].reshape((1,1)), input_t_domain[i,0].reshape((1,1))).reshape((1,1))), dim=0)
                
            PDEloss_tensor.reshape((pde_collocation,1))
            max_stad = torch.sort(PDEloss_tensor, descending=True)
            max_stad = max_stad[0][int(pde_collocation/10)-1,0].item()
            PDEloss_picked = torch.where(PDEloss_tensor>=max_stad, PDEloss_tensor, 0)
            PDEloss_adaptive = torch.sum(PDEloss_picked)/(int(pde_collocation/10))

            
            #Loss computation
            u_IC_loss = lossIC(net, input_x_IC, input_t_IC) #, u_IC_loss_mesh
            mse_IC = u_IC_loss
            
            #Loss based on Boundary Condition (Containing No-Slip and Free-slip)
            mse_BC_u, mse_BC_u_x = lossBdry(net, input_x_BC, input_t_BC) 
            mse_BC = mse_BC_u+ mse_BC_u_x 
            
            #Loss based on PDE
            AC_mse= lossNSpde(net, input_x_domain, input_t_domain) 
            mse_PDE = AC_mse + PDEloss_adaptive
        
            loss =  (mse_BC + mse_IC + mse_PDE )
        
            loss.backward()
            
            
            def closure():
                return loss
            
            #Make Iterative Step
            net.optimizer.step() #net.optimizer.step(closure)
            
            
            # Gradient Norm Clipping
            #torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm = 1000, error_if_nonfinite=False) #note max norm is more art than science
            
            with torch.autograd.no_grad():
                if epoch == 1 or epoch%print_loss == 0:
                    print("\nIteration:", epoch, "\tTotal Loss:", loss.data)
                    #,"\tadaptive IC Loss2: ", mse_IC_a2.item() , "\tadaptive CH PDE Loss2: ", CH_a2.item()
                    print("\tIC Loss: ", mse_IC.item(),
                          "\t BC u Loss: ", mse_BC_u.item(), "\t BC u_x Loss: ", mse_BC_u_x.item(),
                          "\nAC PDE Loss: ", AC_mse.item(), "\tadaptive PDE Loss: ", PDEloss_adaptive.item()
                         ) 
                if epoch%6000 == 0:
                    np.savetxt(f"{ctime}epsilon.txt", epsilon)
                    torch.save(net.state_dict(), f"lr{get_lr(net.optimizer)}_t{final_time}_{ctime}.pt")
        
                

create_network(load=False, loadfile = "lr0.001_t0.2_3_18_11h.pt")


# In[ ]:




