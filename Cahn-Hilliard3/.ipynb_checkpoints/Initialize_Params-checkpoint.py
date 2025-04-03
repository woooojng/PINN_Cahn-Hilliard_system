# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 12:15:55 2024

@author: kevbuck
"""
import torch
import numpy as np
from torch.autograd import Variable
import time
import math
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

#Choose Test Problem
from BM_WZ_CHnet import BM_WZ_CHnet
Net = BM_WZ_CHnet
#torch.autograd.set_detect_anomaly(True)

#Create and Train the network
def create_network(layers, preload = False, preload_name=''):
    
    start = time.time()
    
    
    net = Net(layers)
    net = net.to(device)
    
    if preload == True:
        try:
            net.load_state_dict(torch.load(preload_name, map_location=torch.device('cpu')))
            print('Loaded Successfully')
        except Exception as error:
            print('Loading Failed: ', error)
            pass
        
    time_vec = [0, 0, 0, 0]
    
    #Set final times for running training_loop
    time_slices = np.array([1])
    
    global epsilon #used to track loss
    epsilon = []
    
    print('Training PDE')
    
    iteration_vec = [10000]
    learning_rate_vec = 1e-3 * np.ones_like(iteration_vec)
    #r_vec = [30, 20, 10, 10]
    
    
    for i in range(len(iteration_vec)):
        #Set loop to optimize in progressively smaller learning rates
        print(f'Executing Pass {i+1}')
        iterations = iteration_vec[i]
        learning_rate = learning_rate_vec[i]   
        #r = r_vec[i] #determines sampling distribution decay rate
        r=10 #turn of time-adaptive sampling
        
        training_loop(net, time_slices, iterations, learning_rate, r, record_loss = 100, print_loss = 500)
        torch.save(net.state_dict(), f"CH_Benchmarks_Pass_{i+1}.pt")
        
        time_slices = [time_slices[-1]]
        
        np.savetxt('epsilon.txt', epsilon)
        time_vec[i] = time.time()


    np.savetxt('epsilon.txt', epsilon)
    
    end = time.time()

    print("Total Time:\t", end-start)
    for i in range(len(iteration_vec)):
        if i>0:
            print(f'Pass {i+1} Time: {time_vec[i] - time_vec[i-1]}')
        else:
            print(f'Pass 1 Time: {time_vec[0] - start}')


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def exponential_time_sample(collocation, t_l, t_u, r):
    t_pre = np.random.uniform(0,1, size=(collocation, 1))
    
    t = -np.log(1-t_pre+t_pre*np.exp(-r*(t_u-t_l)))/r
    
    return t



def training_loop(net, time_slices, iterations, learning_rate, r, record_loss = 100, print_loss = 1000):
    global epsilon

    # Domain boundary
    x_l = net.x1_l
    x_u = net.x1_u
    y_l = net.x2_l
    y_u = net.x2_u

    #time starts at lower bound 0, ends at upper bouund updated in slices
    t_l = 0

    # Generate following random numbers for x, y t
    BC_collocation = int(1000) #1000
    IC_collocation = int(5000) #5000
    pde_collocation = int(10000) #10000
    
    #learning rate update
    for g in net.optimizer.param_groups:
        g['lr'] = learning_rate
    
    for final_time in time_slices:
        epoch = 1
         
        with torch.autograd.no_grad():
            print("Current Final Time:", final_time, "Current Learning Rate: ", get_lr(net.optimizer))  
        
        
        #Initialize Adaptive Learning Rates
        a = 1
        b = 1
        a0 = a
        b0 = b
        
        epoch = 0
        while epoch <= iterations:
            # Resetting gradients to zero            
            net.optimizer.zero_grad()    
            
            
            ##Define Colloacation Points with Initial Condition
            x_IC = np.random.uniform(low=x_l, high=x_u, size=(IC_collocation,1))
            y_IC = np.random.uniform(low=y_l, high=y_u, size=(IC_collocation,1))

            input_x_IC = Variable(torch.from_numpy(x_IC).float(), requires_grad=True).to(device)
            input_y_IC = Variable(torch.from_numpy(y_IC).float(), requires_grad=True).to(device)
            
            ##Define Boundary Condition Collocation Points
            x_BC = np.random.uniform(low=x_l, high=x_u, size=(BC_collocation,1))
            y_BC = np.random.uniform(low=y_l, high=y_u, size=(BC_collocation,1))       
            t_BC = np.random.uniform(low=t_l, high=final_time, size=(BC_collocation,1))
    
            input_x_BC= Variable(torch.from_numpy(x_BC).float(), requires_grad=True).to(device)
            input_y_BC = Variable(torch.from_numpy(y_BC).float(), requires_grad=True).to(device)
            input_t_BC = Variable(torch.from_numpy(t_BC).float(), requires_grad=True).to(device)
    
            #Define Inner Domain Collocation Points
            x_domain = np.random.uniform(low= x_l, high=x_u, size=(pde_collocation, 1))
            y_domain = np.random.uniform(low= y_l, high=y_u, size=(pde_collocation, 1))
            t_domain = exponential_time_sample(pde_collocation, t_l, final_time, r)
    
            input_x_domain = Variable(torch.from_numpy(x_domain).float(), requires_grad=True).to(device)
            input_y_domain = Variable(torch.from_numpy(y_domain).float(), requires_grad=True).to(device)
            input_t_domain = Variable(torch.from_numpy(t_domain).float(), requires_grad=True).to(device)
            
            ### Training steps
            ## Compute Loss
            # Loss based on Initial Condition
            mse_IC = net.Initial_Condition_Loss(input_x_IC, input_y_IC)
        
            # Loss based on Boundary Condition (Containing No-Slip and Free-slip)
            mse_BC = net.Boundary_Loss(input_x_BC, input_y_BC, input_t_BC)
        
            # Loss based on PDE
            mse_domain = net.IC_Only_Loss(input_x_domain, input_y_domain, input_t_domain)
            
            # Sum Loss
            loss = a * mse_IC + b * mse_BC +  + mse_domain
            
            ## Compute Gradients
            # Compute Actual Gradients
            loss.backward()
            
            #Gradient Norm Clipping
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm = 100, error_if_nonfinite=False) #note max norm is more art than science
            
            ## Step Optimizer
            net.optimizer.step()
            
            ### Update Epoch
            epoch = epoch + 1
            
            ### Save and Print
            
            #Print Loss every 1000 Epochs
            with torch.autograd.no_grad():
                if epoch%record_loss == 1:
                    epsilon = np.append(epsilon, loss.cpu().detach().numpy())
                if epoch%print_loss == 1:
                    print("Iteration:", epoch, "\tTotal Loss:", loss.item())
                    print("\tLoss: ", loss.item())
                    torch.save(net.state_dict(), "CH_Benchmarks_Pass_Temp.pt")
        
    
layers=[3, 128, 128, 128, 128, 128, 128, 1]
create_network(layers, preload=False, preload_name='CH_Benchmarks_Start.pt')