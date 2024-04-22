import torch
import torch.nn as nn
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import Adam
from torch.nn import BCELoss
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from random import randrange
import random
import os
cwd=os.getcwd()
import scipy.io
from scipy.io import loadmat

from scipy.optimize import minimize

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.linear=nn.Sequential(
            nn.Linear(nz, ngf),
            nn.ReLU(True),
            )
        self.l1_up=nn.ConvTranspose2d(ngf, ngf, 4, 1, 0)
        
        #################################
        # 4 x 4
        #################################
        self.l4_1=nn.ConvTranspose2d(ngf, ngf//2, 3, 1, 1)
        
        self.l4_2=nn.ConvTranspose2d(ngf//2, ngf//2, 3, 1, 1)
        self.l4_batch_2=nn.BatchNorm2d(ngf//2)
        
        self.l4_3=nn.ConvTranspose2d(ngf//2, ngf//2, 3, 1, 1)
        self.l4_batch_3=nn.BatchNorm2d(ngf//2)
        
        self.l4_4=nn.ConvTranspose2d(ngf//2, ngf//2, 3, 1, 1)
        self.l4_batch_4=nn.BatchNorm2d(ngf//2)
        
        self.l4_5=nn.ConvTranspose2d(ngf//2, ngf//2, 3, 1, 1)
        self.l4_batch_5=nn.BatchNorm2d(ngf//2)
        
        self.l4_6=nn.ConvTranspose2d(ngf//2, ngf//2, 3, 1, 1)
        self.l4_batch_6=nn.BatchNorm2d(ngf//2)
        
        self.l4_up=nn.ConvTranspose2d(ngf//2, ngf//2, 4, 2, 1)
        
        #################################
        # 8 x 8
        #################################
        self.l8_1=nn.ConvTranspose2d(ngf//2, ngf//4, 3, 1, 1)
        
        self.l8_2=nn.ConvTranspose2d(ngf//4, ngf//4, 3, 1, 1)
        self.l8_batch_2=nn.BatchNorm2d(ngf//4)
        
        self.l8_3=nn.ConvTranspose2d(ngf//4, ngf//4, 3, 1, 1)
        self.l8_batch_3=nn.BatchNorm2d(ngf//4)
        
        self.l8_4=nn.ConvTranspose2d(ngf//4, ngf//4, 3, 1, 1)
        self.l8_batch_4=nn.BatchNorm2d(ngf//4)
        
        self.l8_5=nn.ConvTranspose2d(ngf//4, ngf//4, 3, 1, 1)
        self.l8_batch_5=nn.BatchNorm2d(ngf//4)
        
        self.l8_6=nn.ConvTranspose2d(ngf//4, ngf//4, 3, 1, 1)
        self.l8_batch_6=nn.BatchNorm2d(ngf//4)
        
        self.l8_up=nn.ConvTranspose2d(ngf//4, ngf//4, 4, 2, 1)
        
        #################################
        # 16 x 16
        #################################
        self.l16_1=nn.ConvTranspose2d(ngf//4, ngf//8, 3, 1, 1)
        
        self.l16_2=nn.ConvTranspose2d(ngf//8, ngf//8, 3, 1, 1)
        self.l16_batch_2=nn.BatchNorm2d(ngf//8)
        
        self.l16_3=nn.ConvTranspose2d(ngf//8, ngf//8, 3, 1, 1)
        self.l16_batch_3=nn.BatchNorm2d(ngf//8)
        
        self.l16_4=nn.ConvTranspose2d(ngf//8, ngf//8, 3, 1, 1)
        self.l16_batch_4=nn.BatchNorm2d(ngf//8)
        
        self.l16_5=nn.ConvTranspose2d(ngf//8, ngf//8, 3, 1, 1)
        self.l16_batch_5=nn.BatchNorm2d(ngf//8)
        
        self.l16_6=nn.ConvTranspose2d(ngf//8, ngf//8, 3, 1, 1)
        self.l16_batch_6=nn.BatchNorm2d(ngf//8)
        
        self.l16_up=nn.ConvTranspose2d(ngf//8, ngf//8, 4, 2, 1)
        
        #################################
        # 32 x 32
        #################################
        self.l32_1=nn.ConvTranspose2d(ngf//8, ngf//16, 3, 1, 1)
        
        self.l32_2=nn.ConvTranspose2d(ngf//16, ngf//16, 3, 1, 1)
        self.l32_batch_2=nn.BatchNorm2d(ngf//16)
        
        self.l32_3=nn.ConvTranspose2d(ngf//16, ngf//16, 3, 1, 1)
        self.l32_batch_3=nn.BatchNorm2d(ngf//16)
        
        self.l32_4=nn.ConvTranspose2d(ngf//16, ngf//16, 3, 1, 1)
        self.l32_batch_4=nn.BatchNorm2d(ngf//16)
        
        self.l32_5=nn.ConvTranspose2d(ngf//16, ngf//16, 3, 1, 1)
        self.l32_batch_5=nn.BatchNorm2d(ngf//16)
        
        self.l32_6=nn.ConvTranspose2d(ngf//16, ngf//16, 3, 1, 1)
        self.l32_batch_6=nn.BatchNorm2d(ngf//16)
        
        self.l32_up=nn.ConvTranspose2d(ngf//16, ngf//16, 4, 2, 1)
        
        #################################
        # 64 x 64
        #################################
        self.l64_1=nn.ConvTranspose2d(ngf//16, ngf//32, 3, 1, 1)
        
        self.l64_2=nn.ConvTranspose2d(ngf//32, ngf//32, 3, 1, 1)
        self.l64_batch_2=nn.BatchNorm2d(ngf//32)
        
        self.l64_3=nn.ConvTranspose2d(ngf//32, ngf//32, 3, 1, 1)
        self.l64_batch_3=nn.BatchNorm2d(ngf//32)
        
        self.l64_4=nn.ConvTranspose2d(ngf//32, ngf//32, 3, 1, 1)
        self.l64_batch_4=nn.BatchNorm2d(ngf//32)
        
        self.l64_5=nn.ConvTranspose2d(ngf//32, ngf//32, 3, 1, 1)
        self.l64_batch_5=nn.BatchNorm2d(ngf//32)
        
        self.l64_6=nn.ConvTranspose2d(ngf//32, ngf//32, 3, 1, 1)
        self.l64_batch_6=nn.BatchNorm2d(ngf//32)
        
        self.l64_final=nn.ConvTranspose2d(ngf//32, 4, 3, 1, 1)

        self.R=nn.ReLU()
    def forward(self, input):
        input=self.linear(input)
        
        x4_1=self.l4_1(self.l1_up(input[:,:,None,None]))
        
        x4_2=self.R(self.l4_batch_2(self.l4_2(x4_1)))
        x4_3=self.R(self.l4_batch_3(self.l4_3(x4_2)))
        x4_3=x4_1+x4_3
        x4_4=self.R(self.l4_batch_4(self.l4_4(x4_3)))
        x4_5=self.R(self.l4_batch_5(self.l4_5(x4_4)))
        x4_6=self.R(self.l4_batch_6(self.l4_6(x4_5)))
        x4_6=x4_4+x4_6
        
        x8_1=self.l8_1(self.l4_up(x4_6))
        
        x8_2=self.R(self.l8_batch_2(self.l8_2(x8_1)))
        x8_3=self.R(self.l8_batch_3(self.l8_3(x8_2)))
        x8_3=x8_1+x8_3
        x8_4=self.R(self.l8_batch_4(self.l8_4(x8_3)))
        x8_5=self.R(self.l8_batch_5(self.l8_5(x8_4)))
        x8_6=self.R(self.l8_batch_6(self.l8_6(x8_5)))
        x8_6=x8_4+x8_6
        
        x16_1=self.l16_1(self.l8_up(x8_6))
        
        x16_2=self.R(self.l16_batch_2(self.l16_2(x16_1)))
        x16_3=self.R(self.l16_batch_3(self.l16_3(x16_2)))
        x16_3=x16_1+x16_3
        x16_4=self.R(self.l16_batch_4(self.l16_4(x16_3)))
        x16_5=self.R(self.l16_batch_5(self.l16_5(x16_4)))
        x16_6=self.R(self.l16_batch_6(self.l16_6(x16_5)))
        x16_6=x16_4+x16_6
        
        x32_1=self.l32_1(self.l16_up(x16_6))
        
        x32_2=self.R(self.l32_batch_2(self.l32_2(x32_1)))
        x32_3=self.R(self.l32_batch_3(self.l32_3(x32_2)))
        x32_3=x32_1+x32_3
        x32_4=self.R(self.l32_batch_4(self.l32_4(x32_3)))
        x32_5=self.R(self.l32_batch_5(self.l32_5(x32_4)))
        x32_6=self.R(self.l32_batch_6(self.l32_6(x32_5)))
        x32_6=x32_4+x32_6
        
        x64_1=self.l64_1(self.l32_up(x32_6))
        
        x64_2=self.R(self.l64_batch_2(self.l64_2(x64_1)))
        x64_3=self.R(self.l64_batch_3(self.l64_3(x64_2)))
        x64_3=x64_1+x64_3
        x64_4=self.R(self.l64_batch_4(self.l64_4(x64_3)))
        x64_5=self.R(self.l64_batch_5(self.l64_5(x64_4)))
        x64_6=self.R(self.l64_batch_6(self.l64_6(x64_5)))
        x64_6=x64_4+x64_6
        
        return self.l64_final(x64_6)

nz=7
ngf=256
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Generator()
model.to(device)
model.load_state_dict(torch.load('./model_1'))
model.eval()

E_min=5
E_max=10
E_choose=[5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10]
N_try=10
N_samples=200
E_idx=0

x = loadmat('./data_E_'+str(E_idx+1)+'.mat')
phi_gather=x.get('phi_gather')

def f_and_grad(x):
    
    w=np.zeros((1,4,64,64))
    w=phi_target
    w=w.astype(np.float32)
    w=torch.tensor(w).to(device)
    
    X_input=(x+5)/25
    temp_x=np.zeros((1, nz))
    temp_x[0,0:nz-1]=X_input
    temp_x[0,nz-1]=(E-E_min)/(E_max-E_min)
    # print(temp_x)
    temp_x=temp_x.astype(np.float32)
    temp_x0=torch.tensor(temp_x,requires_grad=True)
    temp_x=temp_x0.to(device)
    yhat=model(temp_x)
    
    s=torch.sum(torch.abs(w-yhat)**2)
    s.backward()
    g=temp_x0.grad
    g=g.cpu().detach().numpy()
    s=s.cpu().detach().numpy()
    g=g[0,0:nz-1]
    
    return s,g

E=E_choose(E_idx)
R_error=np.zeros(N_samples)
R_V=np.zeros([N_samples,6])
for i in range(0,N_samples):
    
    min_error=10**10
    phi_target=phi_gather[i,:,:,:]
    for j in range(0,N_try):
        
        res=minimize(f_and_grad, np.random.rand(6)*25-5, jac=True)
        res_s,res_g=f_and_grad(res.x)
        if res_s<min_error:
            min_error=res_s
            min_V=res.x
            
    R_error[i]=min_error
    R_V[i,:]=min_V
    print(i,min_error)

scipy.io.savemat(cwd+'/R_error_'+str(E_idx+1)+'.mat', mdict={'R_error': R_error})
scipy.io.savemat(cwd+'/R_V_'+str(E_idx+1)+'.mat', mdict={'R_V': R_V})