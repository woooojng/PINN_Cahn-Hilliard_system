# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 10:52:56 2024

@author: kevbuck
"""

#### Benchmark II: 2D Cahn Hilliard, Seven Circles

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 16:05:35 2023

@author: kevbuck
"""

import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
import sys

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class BM_WZ_CHnet(nn.Module):
    #1 layer N node Neural Network
    def __init__(self, layers):
        super().__init__()
        self.flatten = nn.Flatten()
        
        self.mse_cost_function = torch.nn.MSELoss()
        
        self.layers_u = nn.ModuleList()
        
        #u Network
        self.layers_u.append(nn.Linear(layers[0], layers[1]))
        torch.nn.init.xavier_uniform_(self.layers_u[0].weight)
        
        for i in range(1, len(layers) - 2):
            self.layers_u.append(nn.Linear(layers[i], layers[i+1]))
            torch.nn.init.xavier_uniform_(self.layers_u[i].weight)
        
        self.layers_u.append(nn.Linear(layers[-2], layers[-1]))
        torch.nn.init.xavier_uniform_(self.layers_u[-1].weight)
        '''
        #Mu Network
        self.layers_mu = nn.ModuleList()
        
        self.layers_mu.append(nn.Linear(layers[0], layers[1]))
        torch.nn.init.xavier_uniform_(self.layers_mu[0].weight)
        
        for i in range(1, len(layers) - 2):
            self.layers_mu.append(nn.Linear(layers[i], layers[i+1]))
            torch.nn.init.xavier_uniform_(self.layers_mu[i].weight)
        # Output layer
        self.layers_mu.append(nn.Linear(layers[-2], layers[-1]))
        torch.nn.init.xavier_uniform_(self.layers_mu[-1].weight)
        '''       

        self.optimizer = torch.optim.Adam(self.parameters())
        #self.optimizer = torch.optim.LBFGS(self.parameters())
        
        ## Model Paramters defined here
        self.gamma1 = 10**-6
        self.gamma2 = .01
        
        self.x1_l = -1
        self.x1_u = 1
        #self.x2_l = -1
        #self.x2_u = 1
        self.t0 = 0
        #self.tf = 1
    
    def forward(self, x1, t):
       # relu = torch.nn.ReLU()
        x = torch.cat([x1, t],axis=1)
        x = self.flatten(x)
        
        u = x
        for i in range(len(self.layers_u) - 1):
            temp = self.layers_u[i](u)
            u = torch.tanh(temp)
        u = self.layers_u[-1](u)
        '''
        mu = x
        for i in range(len(self.layers_mu) - 1):
            temp = self.layers_mu[i](mu)
            mu = torch.tanh(temp)
        mu = self.layers_mu[-1](mu)
        '''
        return u #phi, mu
    
    def W(self, x, t):
        u = self(x, t)
        return self.gamma2 *.25*torch.pow((torch.pow(u,2)-1), 2)
    
    def W_exact(self, x):
        u = self.Initial_Condition_u(x)
        return self.gamma2 *.25*torch.pow((torch.pow(u,2)-1), 2)
    
    def W_prime(self, u):
        w_prime = self.gamma2 *u * (u*2 - 1)
        return w_prime
    
    def Initial_Condition_u(self, x):
        
        u_0 = -torch.cos(2*np.pi * x)

        return u_0
    
    def PDE_Loss(self, x, t):
        u = self(x, t)
                    
        u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]  
        
        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]

        mu = self.gamma2 *(u**3 -u) - self.gamma1*u_xx
        
        mu_x = torch.autograd.grad(mu.sum(), x, create_graph=True)[0]
        mu_xx = torch.autograd.grad(mu_x.sum(), x, create_graph=True)[0]

        
        #compute loss
        pde = u_t - mu_xx  #loss for cahn hilliard
        zeros = torch.zeros_like(pde)
        loss = self.mse_cost_function(pde, zeros)
        
        return loss
    '''
    def mu_PDE_Loss(self, x1, x2, t):
        phi, mu = self(x1, x2, t)
        
        phi_x1 = torch.autograd.grad(phi.sum(), x1, create_graph=True)[0]  
        phi_x1x1 = torch.autograd.grad(phi_x1.sum(), x1, create_graph=True)[0]
        phi_x2 = torch.autograd.grad(phi.sum(), x2, create_graph=True)[0]
        phi_x2x2 = torch.autograd.grad(phi_x2.sum(), x2, create_graph=True)[0]
        
        laplacian_phi = phi_x1x1 + phi_x2x2
        
        w_prime = self.W_prime(phi)
        
        pde = mu - self.epsilon**2*laplacian_phi + w_prime
        zeros = torch.zeros_like(pde)
        loss = self.mse_cost_function(pde, zeros)
        
        return loss
    
    def PDE_Loss(self, x1, x2, t):
        #sum total of PDE loss for phi and mu
        return self.mu_PDE_Loss(x1, x2, t) + self.phi_PDE_Loss(x1, x2, t)
    '''
    def Initial_Condition_Loss(self, x1):
        #IC loss for both phi and mu
        t = Variable(torch.zeros_like(x1), requires_grad=True).to(device)
        
        u_pred = self(x1, t)
        u_exact = self.Initial_Condition_u(x1)
        zero = torch.zeros_like(u_exact)
        '''
        phi_exact_x1 = torch.autograd.grad(phi_exact.sum(), x1, create_graph=True)[0]  
        phi_exact_x1x1 = torch.autograd.grad(phi_exact_x1.sum(), x1, create_graph=True)[0]
        phi_exact_x2 = torch.autograd.grad(phi_exact.sum(), x2, create_graph=True)[0]
        phi_exact_x2x2 = torch.autograd.grad(phi_exact_x2.sum(), x2, create_graph=True)[0]
        
        laplacian_phi_exact = phi_exact_x1x1 + phi_exact_x2x2
        
        w_prime = self.W_prime(phi_exact)
        
        mu_exact = self.epsilon**2*laplacian_phi_exact - w_prime
        '''
        #MSE_sum = torch.nn.MSELoss(reduction = 'sum')
        initial_condition_loss = self.mse_cost_function(u_pred, u_exact )/self.mse_cost_function(zero, u_exact )
        
        return initial_condition_loss
    '''
    def IC_Only_Loss(self, x1, x2, t):
        #loss of the entire time domain being equal to initial condition
        #only for initialization of parameters
        phi_pred, mu_pred = self(x1, x2, t)
        phi = self.Initial_Condition_phi(x1, x2)
        
        phi_x1 = torch.autograd.grad(phi.sum(), x1, create_graph=True)[0]  
        phi_x1x1 = torch.autograd.grad(phi_x1.sum(), x1, create_graph=True)[0]
        phi_x2 = torch.autograd.grad(phi.sum(), x2, create_graph=True)[0]
        phi_x2x2 = torch.autograd.grad(phi_x2.sum(), x2, create_graph=True)[0]
        
        laplacian_phi = phi_x1x1 + phi_x2x2
        
        w_prime = self.W_prime(phi)
        
        mu =  -self.epsilon**2*laplacian_phi + w_prime
        
        zero = torch.zeros_like(phi)
        
        initial_condition_loss = self.mse_cost_function(phi_pred, phi)/self.mse_cost_function(phi, zero) + self.mse_cost_function(mu_pred, mu)/self.mse_cost_function(mu, zero)
        
        return initial_condition_loss
    '''
    def Boundary_Loss(self, x1, t):
        
        x1_l_pt = Variable(self.x1_l * torch.ones_like(t), requires_grad=True).to(device)
        x1_u_pt = Variable(self.x1_u * torch.ones_like(t), requires_grad=True).to(device)
        '''
        x2_l_pt = Variable(self.x2_l * torch.ones_like(t), requires_grad=True).to(device)
        x2_u_pt = Variable(self.x2_u * torch.ones_like(t), requires_grad=True).to(device)
        '''
        #Evaluate at the 4 boundaries
        u_lower = self(x1_l_pt, t)
        u_upper = self(x1_u_pt, t)
        '''
        u_left = self(x1_l_pt, x2, t)
        u_right = self(x1_u_pt, x2, t)
        '''
        #homogeneous neumann on phi and mu
        
        u_lower_x = torch.autograd.grad(u_lower.sum(), x1_l_pt, create_graph=True)[0]
        u_upper_x = torch.autograd.grad(u_upper.sum(), x1_u_pt, create_graph=True)[0]
        '''
        mu_left_x1 = torch.autograd.grad(mu_left.sum(), x1_l_pt, create_graph=True)[0]
        mu_right_x1 = torch.autograd.grad(mu_right.sum(), x1_u_pt, create_graph=True)[0]
        '''
        #Periodic Boundary
        u = self.mse_cost_function(u_lower, u_upper)
        u_diff = self.mse_cost_function(u_lower_x, u_upper_x)
        
        #total loss
        #boundary_loss = u_dirichlet + u_neumann
                
        return u, u_diff
    
    # #Begin special methods for accurate circle loss
    
    # def Circle_IC_Loss(self, input_t_circles):
    #     #initial condition loss specifically around the circles
    #     #used only for initialization
    #     num_points = input_t_circles.size()[0]
    #     circles_radius = self.circles_radius
    #     circles_x1 = self.circles_x1
    #     circles_x2 = self.circles_x2
        
    #     x_circle0 = np.random.uniform(low=circles_x1[0]-circles_radius[0], high = circles_x1[0]+circles_radius[0], size=(num_points, 1))
    #     y_circle0 = np.random.uniform(low=circles_x2[0]-circles_radius[0], high = circles_x2[0]+circles_radius[0], size=(num_points, 1))
    
    #     x_circle1 = np.random.uniform(low=circles_x1[1]-circles_radius[1], high = circles_x1[1]+circles_radius[1], size=(num_points, 1))
    #     y_circle1 = np.random.uniform(low=circles_x2[1]-circles_radius[1], high = circles_x2[1]+circles_radius[1], size=(num_points, 1))
        
    #     x_circle2 = np.random.uniform(low=circles_x1[2]-circles_radius[2], high = circles_x1[2]+circles_radius[2], size=(num_points, 1))
    #     y_circle2 = np.random.uniform(low=circles_x2[2]-circles_radius[2], high = circles_x2[2]+circles_radius[2], size=(num_points, 1))
        
    #     x_circle3 = np.random.uniform(low=circles_x1[3]-circles_radius[3], high = circles_x1[3]+circles_radius[3], size=(num_points, 1))
    #     y_circle3 = np.random.uniform(low=circles_x2[3]-circles_radius[3], high = circles_x2[3]+circles_radius[3], size=(num_points, 1))
        
    #     x_circle4 = np.random.uniform(low=circles_x1[4]-circles_radius[4], high = circles_x1[4]+circles_radius[4], size=(num_points, 1))
    #     y_circle4 = np.random.uniform(low=circles_x2[4]-circles_radius[4], high = circles_x2[4]+circles_radius[4], size=(num_points, 1))
        
    #     x_circle5 = np.random.uniform(low=circles_x1[5]-circles_radius[5], high = circles_x1[5]+circles_radius[5], size=(num_points, 1))
    #     y_circle5 = np.random.uniform(low=circles_x2[5]-circles_radius[5], high = circles_x2[5]+circles_radius[5], size=(num_points, 1))
        
    #     x_circle6 = np.random.uniform(low=circles_x1[6]-circles_radius[6], high = circles_x1[6]+circles_radius[6], size=(num_points, 1))
    #     y_circle6 = np.random.uniform(low=circles_x2[6]-circles_radius[6], high = circles_x2[6]+circles_radius[6], size=(num_points, 1))
        
        
    #     input_xc0_domain = Variable(torch.from_numpy(x_circle0).float(), requires_grad=True).to(device)
    #     input_yc0_domain = Variable(torch.from_numpy(y_circle0).float(), requires_grad=True).to(device)
        
    #     input_xc1_domain = Variable(torch.from_numpy(x_circle1).float(), requires_grad=True).to(device)
    #     input_yc1_domain = Variable(torch.from_numpy(y_circle1).float(), requires_grad=True).to(device)
        
    #     input_xc2_domain = Variable(torch.from_numpy(x_circle2).float(), requires_grad=True).to(device)
    #     input_yc2_domain = Variable(torch.from_numpy(y_circle2).float(), requires_grad=True).to(device)
        
    #     input_xc3_domain = Variable(torch.from_numpy(x_circle3).float(), requires_grad=True).to(device)
    #     input_yc3_domain = Variable(torch.from_numpy(y_circle3).float(), requires_grad=True).to(device)
        
    #     input_xc4_domain = Variable(torch.from_numpy(x_circle4).float(), requires_grad=True).to(device)
    #     input_yc4_domain = Variable(torch.from_numpy(y_circle4).float(), requires_grad=True).to(device)
        
    #     input_xc5_domain = Variable(torch.from_numpy(x_circle5).float(), requires_grad=True).to(device)
    #     input_yc5_domain = Variable(torch.from_numpy(y_circle5).float(), requires_grad=True).to(device)
        
    #     input_xc6_domain = Variable(torch.from_numpy(x_circle6).float(), requires_grad=True).to(device)
    #     input_yc6_domain = Variable(torch.from_numpy(y_circle6).float(), requires_grad=True).to(device)
        
    #     circle0_loss = self.IC_Only_Loss(input_xc0_domain, input_yc0_domain, input_t_circles)
    #     circle1_loss = self.IC_Only_Loss(input_xc1_domain, input_yc1_domain, input_t_circles)
    #     circle2_loss = self.IC_Only_Loss(input_xc2_domain, input_yc2_domain, input_t_circles)
    #     circle3_loss = self.IC_Only_Loss(input_xc3_domain, input_yc3_domain, input_t_circles)
    #     circle4_loss = self.IC_Only_Loss(input_xc4_domain, input_yc4_domain, input_t_circles)
    #     circle5_loss = self.IC_Only_Loss(input_xc5_domain, input_yc5_domain, input_t_circles)
    #     circle6_loss = self.IC_Only_Loss(input_xc6_domain, input_yc6_domain, input_t_circles)
        
    #     circle_loss = circle0_loss + circle1_loss + circle2_loss + circle3_loss + circle4_loss + circle5_loss + circle6_loss
        
    #     return circle_loss
    
    # def Circle_PDE_Loss(self, input_t_circles):
    #     #pde loss specifically around the circles
    #     #used only for initialization
        
    #     num_points = input_t_circles.size()[0]
    #     circles_radius = self.circles_radius
    #     circles_x1 = self.circles_x1
    #     circles_x2 = self.circles_x2
        
    #     x_circle0 = np.random.uniform(low=circles_x1[0]-circles_radius[0], high = circles_x1[0]+circles_radius[0], size=(num_points, 1))
    #     y_circle0 = np.random.uniform(low=circles_x2[0]-circles_radius[0], high = circles_x2[0]+circles_radius[0], size=(num_points, 1))
    
    #     x_circle1 = np.random.uniform(low=circles_x1[1]-circles_radius[1], high = circles_x1[1]+circles_radius[1], size=(num_points, 1))
    #     y_circle1 = np.random.uniform(low=circles_x2[1]-circles_radius[1], high = circles_x2[1]+circles_radius[1], size=(num_points, 1))
        
    #     x_circle2 = np.random.uniform(low=circles_x1[2]-circles_radius[2], high = circles_x1[2]+circles_radius[2], size=(num_points, 1))
    #     y_circle2 = np.random.uniform(low=circles_x2[2]-circles_radius[2], high = circles_x2[2]+circles_radius[2], size=(num_points, 1))
        
    #     x_circle3 = np.random.uniform(low=circles_x1[3]-circles_radius[3], high = circles_x1[3]+circles_radius[3], size=(num_points, 1))
    #     y_circle3 = np.random.uniform(low=circles_x2[3]-circles_radius[3], high = circles_x2[3]+circles_radius[3], size=(num_points, 1))
        
    #     x_circle4 = np.random.uniform(low=circles_x1[4]-circles_radius[4], high = circles_x1[4]+circles_radius[4], size=(num_points, 1))
    #     y_circle4 = np.random.uniform(low=circles_x2[4]-circles_radius[4], high = circles_x2[4]+circles_radius[4], size=(num_points, 1))
        
    #     x_circle5 = np.random.uniform(low=circles_x1[5]-circles_radius[5], high = circles_x1[5]+circles_radius[5], size=(num_points, 1))
    #     y_circle5 = np.random.uniform(low=circles_x2[5]-circles_radius[5], high = circles_x2[5]+circles_radius[5], size=(num_points, 1))
        
    #     x_circle6 = np.random.uniform(low=circles_x1[6]-circles_radius[6], high = circles_x1[6]+circles_radius[6], size=(num_points, 1))
    #     y_circle6 = np.random.uniform(low=circles_x2[6]-circles_radius[6], high = circles_x2[6]+circles_radius[6], size=(num_points, 1))
        
    #     input_xc0_domain = Variable(torch.from_numpy(x_circle0).float(), requires_grad=True).to(device)
    #     input_yc0_domain = Variable(torch.from_numpy(y_circle0).float(), requires_grad=True).to(device)
        
    #     input_xc1_domain = Variable(torch.from_numpy(x_circle1).float(), requires_grad=True).to(device)
    #     input_yc1_domain = Variable(torch.from_numpy(y_circle1).float(), requires_grad=True).to(device)
        
    #     input_xc2_domain = Variable(torch.from_numpy(x_circle2).float(), requires_grad=True).to(device)
    #     input_yc2_domain = Variable(torch.from_numpy(y_circle2).float(), requires_grad=True).to(device)
        
    #     input_xc3_domain = Variable(torch.from_numpy(x_circle3).float(), requires_grad=True).to(device)
    #     input_yc3_domain = Variable(torch.from_numpy(y_circle3).float(), requires_grad=True).to(device)
        
    #     input_xc4_domain = Variable(torch.from_numpy(x_circle4).float(), requires_grad=True).to(device)
    #     input_yc4_domain = Variable(torch.from_numpy(y_circle4).float(), requires_grad=True).to(device)
        
    #     input_xc5_domain = Variable(torch.from_numpy(x_circle5).float(), requires_grad=True).to(device)
    #     input_yc5_domain = Variable(torch.from_numpy(y_circle5).float(), requires_grad=True).to(device)
        
    #     input_xc6_domain = Variable(torch.from_numpy(x_circle6).float(), requires_grad=True).to(device)
    #     input_yc6_domain = Variable(torch.from_numpy(y_circle6).float(), requires_grad=True).to(device)
        
    #     circle0_loss = self.PDE_Loss(input_xc0_domain, input_yc0_domain, input_t_circles)
    #     circle1_loss = self.PDE_Loss(input_xc1_domain, input_yc1_domain, input_t_circles)
    #     circle2_loss = self.PDE_Loss(input_xc2_domain, input_yc2_domain, input_t_circles)
    #     circle3_loss = self.PDE_Loss(input_xc3_domain, input_yc3_domain, input_t_circles)
    #     circle4_loss = self.PDE_Loss(input_xc4_domain, input_yc4_domain, input_t_circles)
    #     circle5_loss = self.PDE_Loss(input_xc5_domain, input_yc5_domain, input_t_circles)
    #     circle6_loss = self.PDE_Loss(input_xc6_domain, input_yc6_domain, input_t_circles)
        
    #     circle_loss = circle0_loss + circle1_loss + circle2_loss + circle3_loss + circle4_loss + circle5_loss + circle6_loss
        
    #     return circle_loss