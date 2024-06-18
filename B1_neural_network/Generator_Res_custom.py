import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import os
import random
cwd=os.getcwd()
import scipy.io
from scipy.io import loadmat
import copy

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image_size = 64
alpha=0.01
nz=7
ngf=256
model_index=2
batch_size=128
N_epoch=1000
model = Generator()
model.to(device)

######## Normalized train, validate and test data from 0 to 1 ########
######## Train_data #########
X_input_train=[]
y_input_train=[]
E_input_train=[]
for i in range(1,81):
    x = loadmat(cwd+'/data_'+str(i)+'.mat')
    if len(X_input_train)==0:
        X_input_train=x.get('V_gather')
        y_input_train=x.get('phi_gather')
        E_input_train=x.get('E_gather')
    else:
        X_input_train = np.concatenate((X_input_train,x.get('V_gather')))
        y_input_train = np.concatenate((y_input_train,x.get('phi_gather')))
        E_input_train = np.concatenate((E_input_train,x.get('E_gather')))
E_loss_train=E_input_train
X_input_train=(X_input_train+5)/25
E_input_train=(E_input_train-5)/5
N_train=len(X_input_train)

######## Val_data #########
X_input_val=[]
y_input_val=[]
E_input_val=[]
for i in range(81,91):
    x = loadmat(cwd+'/data_'+str(i)+'.mat')
    if len(X_input_val)==0:
        X_input_val=x.get('V_gather')
        y_input_val=x.get('phi_gather')
        E_input_val=x.get('E_gather')
    else:
        X_input_val = np.concatenate((X_input_val,x.get('V_gather')))
        y_input_val = np.concatenate((y_input_val,x.get('phi_gather')))
        E_input_val = np.concatenate((E_input_val,x.get('E_gather')))
E_loss_val=E_input_val
X_input_val=(X_input_val+5)/25
E_input_val=(E_input_val-5)/5
N_val=len(X_input_val)

######## Test_data #########
X_input_test=[]
y_input_test=[]
E_input_test=[]
for i in range(91,101):
    x = loadmat(cwd+'/data_'+str(i)+'.mat')
    if len(X_input_test)==0:
        X_input_test=x.get('V_gather')
        y_input_test=x.get('phi_gather')
        E_input_test=x.get('E_gather')
    else:
        X_input_test = np.concatenate((X_input_test,x.get('V_gather')))
        y_input_test = np.concatenate((y_input_test,x.get('phi_gather')))
        E_input_test = np.concatenate((E_input_test,x.get('E_gather')))
E_loss_test=E_input_test
X_input_test=(X_input_test+5)/25
E_input_test=(E_input_test-5)/5
N_test=len(X_input_test)

print(N_train,N_val,N_test)
print('Train E',np.max(E_input_train),np.min(E_input_train))
print('Train E loss',np.max(E_loss_train),np.min(E_loss_train))
print('Train V',np.max(X_input_train),np.min(X_input_train))
print('Val E',np.max(E_input_val),np.min(E_input_val))
print('Val E loss',np.max(E_loss_val),np.min(E_loss_val))
print('Val V',np.max(X_input_val),np.min(X_input_val))
print('Test E',np.max(E_input_test),np.min(E_input_test))
print('Test E loss',np.max(E_loss_test),np.min(E_loss_test))
print('Test V',np.max(X_input_test),np.min(X_input_test))
######### enumerate epochs #########

error_train=np.zeros(N_epoch)
error_val=np.zeros(N_epoch)

def generate_batch(N,batch_size):
    
    r_index=random.sample(range(N), N)
    A=[]
    temp=[]
    for i in range(0,N):
        if len(temp)<batch_size:
            temp.append(r_index[i])
        else:
            A.append(temp.copy())
            temp=[]
    if len(temp)!=0:
        A.append(temp)
    return A

######### Train #########
dx=0.031746031746032
dy=0.031746031746032

w_0=np.array([[0,0,0],
     [0,0,0],
     [0,0,0]])
w_dx=np.array([[0,0,0],
     [-1/(2*dx),0,1/(2*dx)],
     [0,0,0]])
w_dy=np.array([[0,-1/(2*dy),0],
      [0,0,0],
      [0,1/(2*dy),0]])
weights=np.array([[w_0,w_0,-w_dy,w_dx],
         [w_0,w_0,-w_dx,-w_dy],
         [w_dy,w_dx,w_0,w_0],
         [-w_dx,w_dy,w_0,w_0]])
weights = weights.astype(np.float32)
weights = torch.tensor(weights).to(device)

def my_custom_loss(output,target,E_batch):
    
    conv = nn.Conv2d(4, 4, 3, bias=False)
    with torch.no_grad():
        conv.weight = nn.Parameter(weights)
    Dirac_check_left = conv(output)
    E_batch=E_batch.repeat([4,image_size-2,image_size-2,1])
    E_batch=torch.permute(E_batch,(3,0,1,2))
    
    Dirac_check_right = output[:,:,1:image_size-1,1:image_size-1]*E_batch
    
    loss1 = torch.mean(torch.abs((output-target))**2)
    loss2 = torch.mean(torch.abs(Dirac_check_left-Dirac_check_right)**2)
    
    return loss1+alpha*loss2

def mse_loss(output,target):
    return torch.mean(torch.abs((output-target))**2)

criterion1 = my_custom_loss
criterion2 = mse_loss
optimizer = Adam(model.parameters(),lr=0.005,weight_decay=1e-5)
best_val_loss=10**10

for epoch in range(N_epoch):
    
    # Train
    model.train()
    sample=generate_batch(N_train,batch_size)
    n_sample_train=len(sample)
    loss_train=0
    for batch_idx in range(0,n_sample_train):
        
        nb=len(sample[batch_idx])
        x_train=np.zeros((nb, nz))
        y_train=np.zeros((nb, 4, image_size, image_size))
        E_train=np.zeros(nb)
        for i in range(0,nb):
            
            x_train[i,0:nz-1]=X_input_train[sample[batch_idx][i],:]
            x_train[i,nz-1]=E_input_train[sample[batch_idx][i]]
            E_train[i]=E_loss_train[sample[batch_idx][i]]
            y_train[i,:,:,:]=y_input_train[sample[batch_idx][i],:,:,:]
            
        x_train=x_train.astype(np.float32)
        x_train=torch.tensor(x_train).to(device)
        y_train=y_train.astype(np.float32)
        y_train=torch.tensor(y_train).to(device)
        E_train=E_train.astype(np.float32)
        E_train=torch.tensor(E_train).to(device)
        
        optimizer.zero_grad()
        yhat = model(x_train)
        loss = criterion1(yhat,y_train,E_train)
        loss.backward()

        optimizer.step()
        loss_train+=loss.cpu().detach().numpy()
   
    # val
    model.eval()
    sample=generate_batch(N_val,batch_size)
    n_sample_val=len(sample)
    loss_val=0
    for batch_idx in range(0,n_sample_val):
        
        nb=len(sample[batch_idx])
        x_val=np.zeros((nb, nz))
        y_val=np.zeros((nb, 4, image_size, image_size))
        for i in range(0,nb):
            
            x_val[i,0:nz-1]=X_input_val[sample[batch_idx][i],:]
            x_val[i,nz-1]=E_input_val[sample[batch_idx][i]]
            y_val[i,:,:,:]=y_input_val[sample[batch_idx][i],:,:,:]
            
        x_val=x_val.astype(np.float32)
        x_val=torch.tensor(x_val).to(device)
        y_val=y_val.astype(np.float32)
        y_val=torch.tensor(y_val).to(device)
        yhat = model(x_val)
        
        loss_val+=criterion2(yhat,y_val).cpu().detach().numpy()
        
    error_train[epoch]=loss_train/n_sample_train
    error_val[epoch]=loss_val/n_sample_val
    print(epoch,error_train[epoch],error_val[epoch])
    if error_val[epoch]<best_val_loss:
        best_val_loss=error_val[epoch]
        best_model=copy.deepcopy(model)
        print("Better val loss", best_val_loss)

# Test the model
best_model.eval()
sample=generate_batch(N_test,batch_size)
n_sample_test=len(sample)
loss_test=0
for batch_idx in range(0,n_sample_test):
        
    nb=len(sample[batch_idx])
    x_test=np.zeros((nb, nz))
    y_test=np.zeros((nb, 4, image_size, image_size))
    for i in range(0,nb):
            
        x_test[i,0:nz-1]=X_input_test[sample[batch_idx][i],:]
        x_test[i,nz-1]=E_input_test[sample[batch_idx][i]]
        y_test[i,:,:,:]=y_input_test[sample[batch_idx][i],:,:,:]
            
    x_test=x_test.astype(np.float32)
    x_test=torch.tensor(x_test).to(device)
    y_test=y_test.astype(np.float32)
    y_test=torch.tensor(y_test).to(device)
    yhat = best_model(x_test)
        
    loss_test+=criterion2(yhat,y_test).cpu().detach().numpy()

loss_test=loss_test/n_sample_test
print("Testing loss", loss_test)

# save the loss and the first 1000 data

scipy.io.savemat(cwd+'/error_train_'+str(model_index)+'.mat', mdict={'error_train': error_train})
scipy.io.savemat(cwd+'/error_val_'+str(model_index)+'.mat', mdict={'error_val': error_val})
scipy.io.savemat(cwd+'/error_test_'+str(model_index)+'.mat', mdict={'loss_test': loss_test})
scipy.io.savemat(cwd+'/alpha_'+str(model_index)+'.mat', mdict={'alpha': alpha})
torch.save(best_model.state_dict(), './model_'+str(model_index))

y_train_pred=np.zeros((1000, 4, 64, 64))
for idx in range(0,1000):
    
    temp_x=np.zeros((1, nz))
    temp_x[0,0:nz-1]=X_input_train[idx,:]
    temp_x[0,nz-1]=E_input_train[idx]
    temp_x=temp_x.astype(np.float32)
    temp_x=torch.tensor(temp_x).to(device)
    yhat=best_model(temp_x)
    y_train_pred[idx,:,:,:]=yhat.cpu().detach().numpy()

scipy.io.savemat(cwd+'/y_train_pred_'+str(model_index)+'.mat', mdict={'y_train_pred': y_train_pred})

y_val_pred=np.zeros((1000, 4, 64, 64))
for idx in range(0,1000):

    temp_x=np.zeros((1, nz))
    temp_x[0,0:nz-1]=X_input_val[idx,:]
    temp_x[0,nz-1]=E_input_val[idx]
    temp_x=temp_x.astype(np.float32)
    temp_x=torch.tensor(temp_x).to(device)
    yhat=best_model(temp_x)
    y_val_pred[idx,:,:,:]=yhat.cpu().detach().numpy()

scipy.io.savemat(cwd+'/y_val_pred_'+str(model_index)+'.mat', mdict={'y_val_pred': y_val_pred})
    
y_test_pred=np.zeros((1000, 4, 64, 64))
for idx in range(0,1000):

    temp_x=np.zeros((1, nz))
    temp_x[0,0:nz-1]=X_input_test[idx,:]
    temp_x[0,nz-1]=E_input_test[idx]
    temp_x=temp_x.astype(np.float32)
    temp_x=torch.tensor(temp_x).to(device)
    yhat=best_model(temp_x)
    y_test_pred[idx,:,:,:]=yhat.cpu().detach().numpy()

scipy.io.savemat(cwd+'/y_test_pred_'+str(model_index)+'.mat', mdict={'y_test_pred': y_test_pred})
