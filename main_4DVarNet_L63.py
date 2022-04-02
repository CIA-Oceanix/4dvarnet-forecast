#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 17:32:46 2021

@author: rfablet
"""
import numpy as np
import matplotlib.pyplot as plt 
import os
#import tensorflow.keras as keras

import time
import copy
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import xarray as xr

from sklearn import decomposition
from netCDF4 import Dataset

import solver as solver_4DVarNet

#os.chdir('/content/drive/My Drive/Colab Notebooks/AnDA')
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
#from AnDA_codes.AnDA_dynamical_models import AnDA_Lorenz_63, AnDA_Lorenz_96
from sklearn.feature_extraction import image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

flagProcess = 0

dimGradSolver = 25
rateDropout = 0.2
DimAE = 10
flagAEType = 'unet2'# 'unet'#'unet2+wc_ode'#'unet' # #'ode' # 
dim_aug_state = 10#10#10#10 #False#

batch_size = 128#2000#

NbTraining = 10000
NbTest     = 2000#256
time_step = 1
dT        = 200
sigNoise  = np.sqrt(2.0)
rateMissingData = (1-1./8.)#0.75#0.95

flagTypeMissData = 2
flagForecast = False#True#
dt_forecast = 55#103#55#
flag_x1_only = False#True #

print('........ Data generation')
flagRandomSeed = 0
if flagRandomSeed == 0:
    print('........ Random seed set to 100')
    np.random.seed(100)
    torch.manual_seed(100)

def AnDA_Lorenz_63(S,t,sigma,rho,beta):
    """ Lorenz-63 dynamical model. """
    x_1 = sigma*(S[1]-S[0]);
    x_2 = S[0]*(rho-S[2])-S[1];
    x_3 = S[0]*S[1] - beta*S[2];
    dS  = np.array([x_1,x_2,x_3]);
    return dS

class GD:
    model = 'Lorenz_63'
    class parameters:
        sigma = 10.0
        rho   = 28.0
        beta  = 8.0/3
    dt_integration = 0.01 # integration time
    dt_states = 1 # number of integeration times between consecutive states (for xt and catalog)
    dt_obs = 8 # number of integration times between consecutive observations (for yo)
    var_obs = np.array([0,1,2]) # indices of the observed variables
    nb_loop_train = 10**2 # size of the catalog
    nb_loop_test = 20000 # size of the true state and noisy observations
    sigma2_catalog = 0.0 # variance of the model error to generate the catalog
    sigma2_obs = 2.0 # variance of the observation error to generate observation

class time_series:
  values = 0.
  time   = 0.
  
flag_load_data =True#  False #  

if flag_load_data == False :
    
    if 1*0 :
        ## data generation: L63 series
        GD = GD()    
        y0 = np.array([8.0,0.0,30.0])
        tt = np.arange(GD.dt_integration,GD.nb_loop_test*GD.dt_integration+0.000001,GD.dt_integration)
        #S = odeint(AnDA_Lorenz_63,x0,np.arange(0,5+0.000001,GD.dt_integration),args=(GD.parameters.sigma,GD.parameters.rho,GD.parameters.beta));
        S = solve_ivp(fun=lambda t,y: AnDA_Lorenz_63(y,t,GD.parameters.sigma,GD.parameters.rho,GD.parameters.beta),t_span=[0.,5+0.000001],y0=y0,first_step=GD.dt_integration,t_eval=np.arange(0,5+0.000001,GD.dt_integration),method='RK45')
        
        y0 = S.y[:,-1];
        S = solve_ivp(fun=lambda t,y: AnDA_Lorenz_63(y,t,GD.parameters.sigma,GD.parameters.rho,GD.parameters.beta),t_span=[GD.dt_integration,GD.nb_loop_test+0.000001],y0=y0,first_step=GD.dt_integration,t_eval=tt,method='RK45')
        S = S.y.transpose()
        
        print( S.shape, flush=True)
        
        ####################################################
        ## Generation of training and test dataset
        ## Extraction of time series of dT time steps            
          
        xt = time_series()
        xt.values = S
        xt.time   = tt
        # extract subsequences
        dataTrainingNoNaN = image.extract_patches_2d(xt.values[0:12000:time_step,:],(dT,3),max_patches=NbTraining)
        dataTestNoNaN     = image.extract_patches_2d(xt.values[15000::time_step,:],(dT,3),max_patches=NbTest)

    else:
        ## data generation: L63 series
        GD = GD()    
        y0 = np.array([8.0,0.0,30.0])
        #S = odeint(AnDA_Lorenz_63,x0,np.arange(0,5+0.000001,GD.dt_integration),args=(GD.parameters.sigma,GD.parameters.rho,GD.parameters.beta));
        
        GD.nb_loop_seq = 15000
        GD.nb_seq = 100
        tt = np.arange(GD.dt_integration,GD.nb_loop_seq*GD.dt_integration+0.000001,GD.dt_integration)
        S0 = solve_ivp(fun=lambda t,y: AnDA_Lorenz_63(y,t,GD.parameters.sigma,GD.parameters.rho,GD.parameters.beta),t_span=[0.,2*GD.nb_seq+5+0.000001],y0=y0,first_step=GD.dt_integration,t_eval=np.arange(0,2*GD.nb_seq+5+0.000001,GD.dt_integration),method='RK45')
        
        for nn in range(0,GD.nb_seq):

            y0 = S0.y[:,500+nn*100]
            S = solve_ivp(fun=lambda t,y: AnDA_Lorenz_63(y,t,GD.parameters.sigma,GD.parameters.rho,GD.parameters.beta),t_span=[GD.dt_integration,GD.nb_loop_seq*GD.dt_integration+0.000001],y0=y0,first_step=GD.dt_integration,t_eval=tt,method='RK45')
            S = S.y.transpose()
              
            ####################################################
            ## Generation of training and test dataset
            ## Extraction of time series of dT time steps            
              
            xt = time_series()
            xt.values = S
            xt.time   = tt
            # extract subsequences
            print('..... (%d) Extract %d+%d patches from a %dx%d sequence '%(nn,int(NbTraining/GD.nb_seq),int(NbTest/GD.nb_seq),S.shape[0],3))
            dataTrainingNoNaN_nn = image.extract_patches_2d(xt.values[0:15000:time_step,:],(dT,3),max_patches=int(NbTraining/GD.nb_seq))
            
            if nn == 0 :
                dataTrainingNoNaN = np.copy( dataTrainingNoNaN_nn )
            else:
                dataTrainingNoNaN = np.concatenate((dataTrainingNoNaN,dataTrainingNoNaN_nn),axis=0)
                
        for nn in range(0,GD.nb_seq):

            y0 = S0.y[:,500+100*GD.nb_seq+nn*100]
            S = solve_ivp(fun=lambda t,y: AnDA_Lorenz_63(y,t,GD.parameters.sigma,GD.parameters.rho,GD.parameters.beta),t_span=[GD.dt_integration,GD.nb_loop_seq*GD.dt_integration+0.000001],y0=y0,first_step=GD.dt_integration,t_eval=tt,method='RK45')
            S = S.y.transpose()
              
            ####################################################
            ## Generation of training and test dataset
            ## Extraction of time series of dT time steps            
              
            xt = time_series()
            xt.values = S
            xt.time   = tt
            # extract subsequences
            print('..... (%d) Extract %d+%d patches from a %dx%d sequence '%(nn,int(NbTraining/GD.nb_seq),int(NbTest/GD.nb_seq),S.shape[0],3))
            dataTestNoNaN_nn     = image.extract_patches_2d(xt.values[:15000:time_step,:],(dT,3),max_patches=int(NbTest/GD.nb_seq))
            
            if nn == 0 :
                dataTestNoNaN = np.copy( dataTestNoNaN_nn )
            else:
                dataTestNoNaN = np.concatenate((dataTestNoNaN,dataTestNoNaN_nn),axis=0)

    flag_save_dataset = True
    if flag_save_dataset == True :
        
        print( dataTrainingNoNaN.shape )
        print( dataTestNoNaN.shape )
        
        xrdata = xr.Dataset( \
            data_vars={'X_train': (('idx_train', 'time', 'l63'), dataTrainingNoNaN), \
                       'X_test': (('idx_test', 'time', 'l63'), dataTestNoNaN) })
#            coords={'idx_train': np.arange(dataTrainingNoNaN.shape[0]),
#                    'idx_test': np.arange(dataTestNoNaN.shape[0]),
#                    'l63': np.arange(3), 
#                    'time': np.arange(dT)})
                    
        xrdata.to_netcdf(path='dataset_L63.nc', mode='w')
else:
    print('.... Load dataset')
    path_l63_dataset = 'dataset_L63.nc'
    ncfile = Dataset(path_l63_dataset,"r")
    dataTrainingNoNaN = ncfile.variables['X_train'][:]
    dataTestNoNaN = ncfile.variables['X_test'][:]
    
    dataTrainingNoNaN = dataTrainingNoNaN[:NbTraining,:,:]
    dataTestNoNaN = dataTestNoNaN[:NbTest,:,:]
    #dataTestNoNaN = dataTestNoNaN[:128,:,:]  
    
# create missing data
if flagTypeMissData == 0:
    print('..... Observation pattern: Random sampling of osberved L63 components')
    indRand         = np.random.permutation(dataTrainingNoNaN.shape[0]*dataTrainingNoNaN.shape[1]*dataTrainingNoNaN.shape[2])
    indRand         = indRand[0:int(rateMissingData*len(indRand))]
    dataTraining    = np.copy(dataTrainingNoNaN).reshape((dataTrainingNoNaN.shape[0]*dataTrainingNoNaN.shape[1]*dataTrainingNoNaN.shape[2],1))
    dataTraining[indRand] = float('nan')
    dataTraining    = np.reshape(dataTraining,(dataTrainingNoNaN.shape[0],dataTrainingNoNaN.shape[1],dataTrainingNoNaN.shape[2]))
    
    indRand         = np.random.permutation(dataTestNoNaN.shape[0]*dataTestNoNaN.shape[1]*dataTestNoNaN.shape[2])
    indRand         = indRand[0:int(rateMissingData*len(indRand))]
    dataTest        = np.copy(dataTestNoNaN).reshape((dataTestNoNaN.shape[0]*dataTestNoNaN.shape[1]*dataTestNoNaN.shape[2],1))
    dataTest[indRand] = float('nan')
    dataTest          = np.reshape(dataTest,(dataTestNoNaN.shape[0],dataTestNoNaN.shape[1],dataTestNoNaN.shape[2]))

    genSuffixObs    = '_ObsRnd_%02d_%02d'%(100*rateMissingData,10*sigNoise**2)
elif flagTypeMissData == 2:
    print('..... Observation pattern: Only the first L63 component is osberved')
    time_step_obs   = int(1./(1.-rateMissingData))
    dataTraining    = np.zeros((dataTrainingNoNaN.shape))
    dataTraining[:] = float('nan')
    dataTraining[:,::time_step_obs,0] = dataTrainingNoNaN[:,::time_step_obs,0]
    
    dataTest    = np.zeros((dataTestNoNaN.shape))
    dataTest[:] = float('nan')
    dataTest[:,::time_step_obs,0] = dataTestNoNaN[:,::time_step_obs,0]

    genSuffixObs    = '_ObsDim0_%02d_%02d'%(100*rateMissingData,10*sigNoise**2)
   
else:
    print('..... Observation pattern: All  L63 components osberved')
    time_step_obs   = int(1./(1.-rateMissingData))
    dataTraining    = np.zeros((dataTrainingNoNaN.shape))
    dataTraining[:] = float('nan')
    dataTraining[:,::time_step_obs,:] = dataTrainingNoNaN[:,::time_step_obs,:]
    
    dataTest    = np.zeros((dataTestNoNaN.shape))
    dataTest[:] = float('nan')
    dataTest[:,::time_step_obs,:] = dataTestNoNaN[:,::time_step_obs,:]

    genSuffixObs    = '_ObsSub_%02d_%02d'%(100*rateMissingData,10*sigNoise**2)
    
# set to NaN patch boundaries
dataTraining[:,0:10,:] =  float('nan')
dataTest[:,0:10,:]     =  float('nan')
dataTraining[:,dT-10:dT,:] =  float('nan')
dataTest[:,dT-10:dT,:]     =  float('nan')

# 
if flagForecast == True :
    dataTraining[:,dT-dt_forecast:,:] =  float('nan')
    dataTest[:,dT-dt_forecast:,:]     =  float('nan')
    
    print(dataTraining.shape)
    print(dataTraining[10,dT-dt_forecast-5:dT-dt_forecast,0])

# mask for NaN
maskTraining = (dataTraining == dataTraining).astype('float')
maskTest     = ( dataTest    ==  dataTest   ).astype('float')

dataTraining = np.nan_to_num(dataTraining)
dataTest     = np.nan_to_num(dataTest)

# Permutation to have channel as #1 component
dataTraining      = np.moveaxis(dataTraining,-1,1)
maskTraining      = np.moveaxis(maskTraining,-1,1)
dataTrainingNoNaN = np.moveaxis(dataTrainingNoNaN,-1,1)

dataTest      = np.moveaxis(dataTest,-1,1)
maskTest      = np.moveaxis(maskTest,-1,1)
dataTestNoNaN = np.moveaxis(dataTestNoNaN,-1,1)

# set to NaN patch boundaries
#dataTraining[:,0:5,:] =  dataTrainingNoNaN[:,0:5,:]
#dataTest[:,0:5,:]     =  dataTestNoNaN[:,0:5,:]

############################################
## raw data
X_train         = dataTrainingNoNaN
X_train_missing = dataTraining
mask_train      = maskTraining

X_test         = dataTestNoNaN
X_test_missing = dataTest
mask_test      = maskTest

############################################
## normalized data
#meanTr          = np.mean(X_train_missing[:]) / np.mean(mask_train) 
#stdTr           = np.sqrt( np.mean( (X_train_missing-meanTr)**2 ) / np.mean(mask_train) )

meanTr          = np.mean(X_train[:],) 
stdTr           = np.sqrt( np.mean( (X_train-meanTr)**2 ) )

x_train_missing = ( X_train_missing - meanTr ) / stdTr
x_test_missing  = ( X_test_missing - meanTr ) / stdTr

x_train = (X_train - meanTr) / stdTr
x_test  = (X_test - meanTr) / stdTr

print('.... MeanTr = %.3f --- StdTr = %.3f '%(meanTr,stdTr))

# Generate noisy observsation
X_train_obs = X_train_missing + sigNoise * maskTraining * np.random.randn(X_train_missing.shape[0],X_train_missing.shape[1],X_train_missing.shape[2])
X_test_obs  = X_test_missing  + sigNoise * maskTest * np.random.randn(X_test_missing.shape[0],X_test_missing.shape[1],X_test_missing.shape[2])

x_train_obs = (X_train_obs - meanTr) / stdTr
x_test_obs  = (X_test_obs - meanTr) / stdTr

print('..... Training dataset: %dx%dx%d'%(x_train.shape[0],x_train.shape[1],x_train.shape[2]))
print('..... Test dataset    : %dx%dx%d'%(x_test.shape[0],x_test.shape[1],x_test.shape[2]))

import scipy
# Initialization
flagInit = 1
mx_train = np.sum( np.sum( X_train , axis = 2 ) , axis = 0 ) / (X_train.shape[0]*X_train.shape[2])

if flagInit == 0: 
  X_train_Init = mask_train * X_train_obs + (1. - mask_train) * (np.zeros(X_train_missing.shape) + meanTr)
  X_test_Init  = mask_test * X_test_obs + (1. - mask_test) * (np.zeros(X_test_missing.shape) + meanTr)
else:
  X_train_Init = np.zeros(X_train.shape)
  for ii in range(0,X_train.shape[0]):
    # Initial linear interpolation for each component
    XInit = np.zeros((X_train.shape[1],X_train.shape[2]))

    for kk in range(0,3):
      indt  = np.where( mask_train[ii,kk,:] == 1.0 )[0]
      indt_ = np.where( mask_train[ii,kk,:] == 0.0 )[0]

      if len(indt) > 1:
        indt_[ np.where( indt_ < np.min(indt)) ] = np.min(indt)
        indt_[ np.where( indt_ > np.max(indt)) ] = np.max(indt)
        fkk = scipy.interpolate.interp1d(indt, X_train_obs[ii,kk,indt])
        XInit[kk,indt]  = X_train_obs[ii,kk,indt]
        XInit[kk,indt_] = fkk(indt_)
      else:
        XInit[kk,:] = XInit[kk,:] +  mx_train[kk]

    X_train_Init[ii,:,:] = XInit

  X_test_Init = np.zeros(X_test.shape)
  for ii in range(0,X_test.shape[0]):
    # Initial linear interpolation for each component
    XInit = np.zeros((X_test.shape[1],X_test.shape[2]))

    for kk in range(0,3):
      indt  = np.where( mask_test[ii,kk,:] == 1.0 )[0]
      indt_ = np.where( mask_test[ii,kk,:] == 0.0 )[0]

      if len(indt) > 1:
        indt_[ np.where( indt_ < np.min(indt)) ] = np.min(indt)
        indt_[ np.where( indt_ > np.max(indt)) ] = np.max(indt)
        fkk = scipy.interpolate.interp1d(indt, X_test_obs[ii,kk,indt])
        XInit[kk,indt]  = X_test_obs[ii,kk,indt]
        XInit[kk,indt_] = fkk(indt_)
      else:
        XInit[kk,:] = XInit[kk,:] +  mx_train[kk]

    X_test_Init[ii,:,:] = XInit

x_train_Init = ( X_train_Init - meanTr ) / stdTr
x_test_Init = ( X_test_Init - meanTr ) / stdTr

# reshape to 2D tensors
x_train = x_train.reshape((-1,3,dT,1))
mask_train = mask_train.reshape((-1,3,dT,1))
x_train_Init = x_train_Init.reshape((-1,3,dT,1))
x_train_obs = x_train_obs.reshape((-1,3,dT,1))

x_test = x_test.reshape((-1,3,dT,1))
mask_test = mask_test.reshape((-1,3,dT,1))
x_test_Init = x_test_Init.reshape((-1,3,dT,1))
x_test_obs = x_test_obs.reshape((-1,3,dT,1))

print('..... Training dataset: %dx%dx%dx%d'%(x_train.shape[0],x_train.shape[1],x_train.shape[2],x_train.shape[3]))
print('..... Test dataset    : %dx%dx%dx%d'%(x_test.shape[0],x_test.shape[1],x_test.shape[2],x_test.shape[3]))


################################################
## dataloader
idx_val = x_train.shape[0]-500

training_dataset     = torch.utils.data.TensorDataset(torch.Tensor(np.arange(0,idx_val)),torch.Tensor(x_train_Init[:idx_val:,:,:,:]),torch.Tensor(x_train_obs[:idx_val:,:,:,:]),torch.Tensor(mask_train[:idx_val:,:,:,:]),torch.Tensor(x_train[:idx_val:,:,:,:])) # create your datset
val_dataset         = torch.utils.data.TensorDataset(torch.Tensor(np.arange(0,500)),torch.Tensor(x_train_Init[idx_val::,:,:,:]),torch.Tensor(x_train_obs[idx_val::,:,:,:]),torch.Tensor(mask_train[idx_val::,:,:,:]),torch.Tensor(x_train[idx_val::,:,:,:])) # create your datset
test_dataset         = torch.utils.data.TensorDataset(torch.Tensor(np.arange(0,x_test_Init.shape[0])),torch.Tensor(x_test_Init),torch.Tensor(x_test_obs),torch.Tensor(mask_test),torch.Tensor(x_test)) # create your datset

dataloaders = {
    'train': torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
    'val': torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),
    'test': torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),
}            
dataset_sizes = {'train': len(training_dataset), 'val': len(val_dataset), 'test': len(test_dataset)}


print('........ Define AE architecture')
shapeData  = np.array(x_train.shape[1:])
if dim_aug_state > 0 :
    shapeData[0] += dim_aug_state
# freeze all ode parameters

if flagAEType == 'ode': ## AE using ode_L63

    class Phi_r(torch.nn.Module):
        def __init__(self):
              super(Phi_r, self).__init__()
              self.sigma = torch.nn.Parameter(torch.Tensor([np.random.randn()]))
              self.rho    = torch.nn.Parameter(torch.Tensor([np.random.randn()]))
              self.beta   = torch.nn.Parameter(torch.Tensor([np.random.randn()]))

              self.sigma  = torch.nn.Parameter(torch.Tensor([10.]))
              self.rho    = torch.nn.Parameter(torch.Tensor([28.]))
              self.beta   = torch.nn.Parameter(torch.Tensor([8./3.]))

              self.dt        = 0.01
              self.IntScheme = 'rk4'
              self.stdTr     = stdTr
              self.meanTr    = meanTr                      
        def _odeL63(self, xin):
            x1  = xin[:,0,:]
            x2  = xin[:,1,:]
            x3  = xin[:,2,:]
            
            dx_1 = (self.sigma*(x2-x1)).view(-1,1,xin.size(2))
            dx_2 = (x1*(self.rho-x3)-x2).view(-1,1,xin.size(2))
            dx_3 = (x1*x2 - self.beta*x3).view(-1,1,xin.size(2))
            
            dpred = torch.cat((dx_1,dx_2,dx_3),dim=1)
            return dpred

        def _EulerSolver(self, x):
            return x + self.dt * self._odeL63(x)

        def _RK4Solver(self, x):
            k1 = self._odeL63(x)
            x2 = x + 0.5 * self.dt * k1
            k2 = self._odeL63(x2)
          
            x3 = x + 0.5 * self.dt * k2
            k3 = self._odeL63(x3)
              
            x4 = x + self.dt * k3
            k4 = self._odeL63(x4)

            return x + self.dt * (k1+2.*k2+2.*k3+k4)/6.
      
        def forward(self, x):
            X = self.stdTr * x.view(-1,x.size(1),x.size(2))
            X = X + self.meanTr
            
            if self.IntScheme == 'euler':
                xpred = self._EulerSolver( X[:,:,0:x.size(2)-1] )
            else:
                xpred = self._RK4Solver( X[:,:,0:x.size(2)-1] )

            xpred = xpred - self.meanTr
            xpred = xpred / self.stdTr

            xnew  = torch.cat((x[:,:,0].view(-1,x.size(1),1),xpred),dim=2)
            
            xnew = xnew.view(-1,x.size(1),x.size(2),1)
            
            return xnew

elif flagAEType == 'unet': ## Conv model with no use of the central point
  dW = 5
  class Phi_r(torch.nn.Module):
      def __init__(self):
          super(Phi_r, self).__init__()
          self.pool1  = torch.nn.AvgPool2d((4,1))
          #self.conv1  = ConstrainedConv2d(shapeData[0],2*shapeData[0]*DimAE,(2*dW+1,1),padding=(dW,0),bias=False)
          self.conv1  = torch.nn.Conv2d(shapeData[0],2*shapeData[0]*DimAE,(2*dW+1,1),padding=(dW,0),bias=False)
          self.conv2  = torch.nn.Conv2d(2*shapeData[0]*DimAE,shapeData[0]*DimAE,1,padding=0,bias=False)
          
          self.conv21 = torch.nn.Conv2d(shapeData[0]*DimAE,shapeData[0]*DimAE,1,padding=0,bias=False)
          self.conv22 = torch.nn.Conv2d(shapeData[0]*DimAE,shapeData[0]*DimAE,1,padding=0,bias=False)
          self.conv23 = torch.nn.Conv2d(shapeData[0]*DimAE,shapeData[0]*DimAE,1,padding=0,bias=False)
          self.conv3  = torch.nn.Conv2d(2*shapeData[0]*DimAE,shapeData[0]*DimAE,1,padding=0,bias=False)
          #self.conv4 = torch.nn.Conv1d(4*shapeData[0]*DimAE,8*shapeData[0]*DimAE,1,padding=0,bias=False)

          self.conv2Tr = torch.nn.ConvTranspose2d(shapeData[0]*DimAE,shapeData[0],(4,1),stride=(4,1),bias=False)          
          #self.conv5 = torch.nn.Conv1d(2*shapeData[0]*DimAE,2*shapeData[0]*DimAE,3,padding=1,bias=False)
          #self.conv6 = torch.nn.Conv1d(2*shapeData[0]*DimAE,shapeData[0],1,padding=0,bias=False)
          #self.conv6 = torch.nn.Conv1d(16*shapeData[0]*DimAE,shapeData[0],3,padding=1,bias=False)

          #self.convHR1  = ConstrainedConv2d(shapeData[0],2*shapeData[0]*DimAE,(2*dW+1,1),padding=(dW,0),bias=False)
          #self.convHR1  = ConstrainedConv2d(shapeData[0],2*shapeData[0]*DimAE,(2*dW+1,1),padding=(dW,0),bias=False)
          self.convHR1  = torch.nn.Conv2d(shapeData[0],2*shapeData[0]*DimAE,(2*dW+1,1),padding=(dW,0),bias=False)
          self.convHR2  = torch.nn.Conv2d(2*shapeData[0]*DimAE,shapeData[0]*DimAE,1,padding=0,bias=False)
          
          self.convHR21 = torch.nn.Conv2d(shapeData[0]*DimAE,shapeData[0]*DimAE,1,padding=0,bias=False)
          self.convHR22 = torch.nn.Conv2d(shapeData[0]*DimAE,shapeData[0]*DimAE,1,padding=0,bias=False)
          self.convHR23 = torch.nn.Conv2d(shapeData[0]*DimAE,shapeData[0]*DimAE,1,padding=0,bias=False)
          self.convHR3  = torch.nn.Conv2d(2*shapeData[0]*DimAE,shapeData[0],1,padding=0,bias=False)

      def forward(self, xinp):
          #x = self.fc1( torch.nn.Flatten(x) )
          #x = self.pool1( xinp )
          x = self.pool1( xinp )
          x = self.conv1( x )
          x = self.conv2( F.relu(x) )
          x = torch.cat((self.conv21(x), self.conv22(x) * self.conv23(x)),dim=1)
          x = self.conv3( x )
          x = self.conv2Tr( x )
          #x = self.conv5( F.relu(x) )
          #x = self.conv6( F.relu(x) )
          
          xHR = self.convHR1( xinp )
          xHR = self.convHR2( F.relu(xHR) )
          xHR = torch.cat((self.convHR21(xHR), self.convHR22(xHR) * self.convHR23(xHR)),dim=1)
          xHR = self.convHR3( xHR )
          
          x   = torch.add(x,1.,xHR)
          
          x = x.view(-1,shapeData[0],shapeData[1],1)
          return x
elif flagAEType == 'unet2': ## Conv model with no use of the central point
  dW = 5
  class Phi_r(torch.nn.Module):
      def __init__(self):
          super(Phi_r, self).__init__()
          self.pool1  = torch.nn.AvgPool2d((5,1))
          self.pool2  = torch.nn.AvgPool2d((2,1))
          #self.conv1  = ConstrainedConv2d(shapeData[0],2*shapeData[0]*DimAE,(2*dW+1,1),padding=(dW,0),bias=False)

          self.conv0  = torch.nn.Conv2d(shapeData[0],shapeData[0]*DimAE,(2*dW+1,1),padding=(dW,0),bias=False)

          self.conv21  = torch.nn.Conv2d(shapeData[0]*DimAE,2*shapeData[0]*DimAE,(2*dW+1,1),padding=(dW,0),bias=False)
          self.conv22  = torch.nn.Conv2d(2*shapeData[0]*DimAE,shapeData[0]*DimAE,1,padding=0,bias=False)          
          self.conv221 = torch.nn.Conv2d(shapeData[0]*DimAE,shapeData[0]*DimAE,1,padding=0,bias=False)
          self.conv222 = torch.nn.Conv2d(shapeData[0]*DimAE,shapeData[0]*DimAE,1,padding=0,bias=False)
          self.conv223 = torch.nn.Conv2d(shapeData[0]*DimAE,shapeData[0]*DimAE,1,padding=0,bias=False)
          self.conv23  = torch.nn.Conv2d(2*shapeData[0]*DimAE,shapeData[0]*DimAE,1,padding=0,bias=False)
          self.conv2Tr = torch.nn.ConvTranspose2d(shapeData[0]*DimAE,shapeData[0]*DimAE,(2,1),stride=(2,1),bias=False)          

          self.conv11  = torch.nn.Conv2d(shapeData[0]*DimAE,2*shapeData[0]*DimAE,(2*dW+1,1),padding=(dW,0),bias=False)
          self.conv12  = torch.nn.Conv2d(2*shapeData[0]*DimAE,shapeData[0]*DimAE,1,padding=0,bias=False)          
          self.conv121 = torch.nn.Conv2d(shapeData[0]*DimAE,shapeData[0]*DimAE,1,padding=0,bias=False)
          self.conv122 = torch.nn.Conv2d(shapeData[0]*DimAE,shapeData[0]*DimAE,1,padding=0,bias=False)
          self.conv123 = torch.nn.Conv2d(shapeData[0]*DimAE,shapeData[0]*DimAE,1,padding=0,bias=False)
          self.conv13  = torch.nn.Conv2d(2*shapeData[0]*DimAE,shapeData[0]*DimAE,1,padding=0,bias=False)
          self.conv1Tr = torch.nn.ConvTranspose2d(shapeData[0]*DimAE,shapeData[0],(5,1),stride=(5,1),bias=False)          

          self.conv01  = torch.nn.Conv2d(shapeData[0]*DimAE,2*shapeData[0]*DimAE,(2*dW+1,1),padding=(dW,0),bias=False)
          self.conv02  = torch.nn.Conv2d(2*shapeData[0]*DimAE,shapeData[0]*DimAE,1,padding=0,bias=False)          
          self.conv021 = torch.nn.Conv2d(shapeData[0]*DimAE,shapeData[0]*DimAE,1,padding=0,bias=False)
          self.conv022 = torch.nn.Conv2d(shapeData[0]*DimAE,shapeData[0]*DimAE,1,padding=0,bias=False)
          self.conv023 = torch.nn.Conv2d(shapeData[0]*DimAE,shapeData[0]*DimAE,1,padding=0,bias=False)
          self.conv03  = torch.nn.Conv2d(2*shapeData[0]*DimAE,shapeData[0],1,padding=0,bias=False)

      def forward(self, xinp):
          #x = self.fc1( torch.nn.Flatten(x) )
          #x = self.pool1( xinp )
          
          xf = self.conv0( xinp )
          
          x2 = self.pool2( self.pool1( xf ) )
          x2 = self.conv21( x2 )
          x2 = self.conv22( F.relu(x2) )
          x2 = torch.cat((self.conv221(x2), self.conv222(x2) * self.conv223(x2)),dim=1)
          x2 = self.conv23( x2 )
          x2 = self.conv2Tr( x2 )

          x1 = self.pool1( xf )
          x1 = self.conv11( x1 )
          x1 = self.conv12( F.relu(x1) )
          x1 = torch.cat((self.conv121(x1), self.conv122(x1) * self.conv123(x1)),dim=1)
          x1 = self.conv13( x1 )
          x1 = self.conv1Tr( x1 + x2 )
                   

          x0 = self.conv01( xf )
          x0 = self.conv02( F.relu(x0) )
          x0 = torch.cat((self.conv021(x0), self.conv022(x0) * self.conv023(x0)),dim=1)
          x0 = self.conv03( x0 )
           
          x   = x1 + x0
          
          x = x.view(-1,shapeData[0],shapeData[1],1)
          return x
elif flagAEType == 'unet2+sc_ode': ## Conv model with no use of the central point
  dW = 5

  class Odenet_l63(torch.nn.Module):
        def __init__(self):
              super(Odenet_l63, self).__init__()
              self.sigma = torch.nn.Parameter(torch.Tensor([np.random.randn()]))
              self.rho    = torch.nn.Parameter(torch.Tensor([np.random.randn()]))
              self.beta   = torch.nn.Parameter(torch.Tensor([np.random.randn()]))

              self.sigma  = torch.nn.Parameter(torch.Tensor([10.]))
              self.rho    = torch.nn.Parameter(torch.Tensor([28.]))
              self.beta   = torch.nn.Parameter(torch.Tensor([8./3.]))

              self.dt        = 0.01
              self.IntScheme = 'rk4'
              self.stdTr     = stdTr
              self.meanTr    = meanTr                      
        def _odeL63(self, xin):
            x1  = xin[:,0,:]
            x2  = xin[:,1,:]
            x3  = xin[:,2,:]
            
            dx_1 = (self.sigma*(x2-x1)).view(-1,1,xin.size(2))
            dx_2 = (x1*(self.rho-x3)-x2).view(-1,1,xin.size(2))
            dx_3 = (x1*x2 - self.beta*x3).view(-1,1,xin.size(2))
            
            dpred = torch.cat((dx_1,dx_2,dx_3),dim=1)
            return dpred

        def _EulerSolver(self, x):
            return x + self.dt * self._odeL63(x)

        def _RK4Solver(self, x):
            k1 = self._odeL63(x)
            x2 = x + 0.5 * self.dt * k1
            k2 = self._odeL63(x2)
          
            x3 = x + 0.5 * self.dt * k2
            k3 = self._odeL63(x3)
              
            x4 = x + self.dt * k3
            k4 = self._odeL63(x4)

            return x + self.dt * (k1+2.*k2+2.*k3+k4)/6.
      
        def forward(self, x0 , dt):
            X = self.stdTr * x0.view(-1,x0.size(1),x0.size(2))
            X = X + self.meanTr
             
            xnew = X.view(-1,X.size(1),1)
            xpred = 1. * xnew
            for t in range(1,dt):
                if self.IntScheme == 'euler':
                    xpred = self._EulerSolver( xpred )
                else:
                    xpred = self._RK4Solver( xpred )
                xnew  = torch.cat((xnew,xpred),dim=2)

            xnew = xnew - self.meanTr
            xnew = xnew / self.stdTr

            xnew = xnew.view(-1,x0.size(1),dt,1)
            
            return xnew

  class Phi_r(torch.nn.Module):
      def __init__(self):
          super(Phi_r, self).__init__()
          self.pool1  = torch.nn.AvgPool2d((5,1))
          self.pool2  = torch.nn.AvgPool2d((2,1))
          #self.conv1  = ConstrainedConv2d(shapeData[0],2*shapeData[0]*DimAE,(2*dW+1,1),padding=(dW,0),bias=False)

          self.conv0  = torch.nn.Conv2d(shapeData[0],shapeData[0]*DimAE,(2*dW+1,1),padding=(dW,0),bias=False)

          self.conv21  = torch.nn.Conv2d(shapeData[0]*DimAE,2*shapeData[0]*DimAE,(2*dW+1,1),padding=(dW,0),bias=False)
          self.conv22  = torch.nn.Conv2d(2*shapeData[0]*DimAE,shapeData[0]*DimAE,1,padding=0,bias=False)          
          self.conv221 = torch.nn.Conv2d(shapeData[0]*DimAE,shapeData[0]*DimAE,1,padding=0,bias=False)
          self.conv222 = torch.nn.Conv2d(shapeData[0]*DimAE,shapeData[0]*DimAE,1,padding=0,bias=False)
          self.conv223 = torch.nn.Conv2d(shapeData[0]*DimAE,shapeData[0]*DimAE,1,padding=0,bias=False)
          self.conv23  = torch.nn.Conv2d(2*shapeData[0]*DimAE,shapeData[0]*DimAE,1,padding=0,bias=False)
          self.conv2Tr = torch.nn.ConvTranspose2d(shapeData[0]*DimAE,shapeData[0]*DimAE,(2,1),stride=(2,1),bias=False)          

          self.conv11  = torch.nn.Conv2d(shapeData[0]*DimAE,2*shapeData[0]*DimAE,(2*dW+1,1),padding=(dW,0),bias=False)
          self.conv12  = torch.nn.Conv2d(2*shapeData[0]*DimAE,shapeData[0]*DimAE,1,padding=0,bias=False)          
          self.conv121 = torch.nn.Conv2d(shapeData[0]*DimAE,shapeData[0]*DimAE,1,padding=0,bias=False)
          self.conv122 = torch.nn.Conv2d(shapeData[0]*DimAE,shapeData[0]*DimAE,1,padding=0,bias=False)
          self.conv123 = torch.nn.Conv2d(shapeData[0]*DimAE,shapeData[0]*DimAE,1,padding=0,bias=False)
          self.conv13  = torch.nn.Conv2d(2*shapeData[0]*DimAE,shapeData[0]*DimAE,1,padding=0,bias=False)
          self.conv1Tr = torch.nn.ConvTranspose2d(shapeData[0]*DimAE,shapeData[0],(5,1),stride=(5,1),bias=False)          

          self.conv01  = torch.nn.Conv2d(shapeData[0]*DimAE,2*shapeData[0]*DimAE,(2*dW+1,1),padding=(dW,0),bias=False)
          self.conv02  = torch.nn.Conv2d(2*shapeData[0]*DimAE,shapeData[0]*DimAE,1,padding=0,bias=False)          
          self.conv021 = torch.nn.Conv2d(shapeData[0]*DimAE,shapeData[0]*DimAE,1,padding=0,bias=False)
          self.conv022 = torch.nn.Conv2d(shapeData[0]*DimAE,shapeData[0]*DimAE,1,padding=0,bias=False)
          self.conv023 = torch.nn.Conv2d(shapeData[0]*DimAE,shapeData[0]*DimAE,1,padding=0,bias=False)
          self.conv03  = torch.nn.Conv2d(2*shapeData[0]*DimAE,shapeData[0],1,padding=0,bias=False)

          self.ode_l63 = Odenet_l63()
          self.dt_forecast = dt_forecast          

      def forward(self, xinp):
          #x = self.fc1( torch.nn.Flatten(x) )
          #x = self.pool1( xinp )
          
          # assimilation component
          xf = self.conv0( xinp )
          
          x2 = self.pool2( self.pool1( xf ) )
          x2 = self.conv21( x2 )
          x2 = self.conv22( F.relu(x2) )
          x2 = torch.cat((self.conv221(x2), self.conv222(x2) * self.conv223(x2)),dim=1)
          x2 = self.conv23( x2 )
          x2 = self.conv2Tr( x2 )

          x1 = self.pool1( xf )
          x1 = self.conv11( x1 )
          x1 = self.conv12( F.relu(x1) )
          x1 = torch.cat((self.conv121(x1), self.conv122(x1) * self.conv123(x1)),dim=1)
          x1 = self.conv13( x1 )
          x1 = self.conv1Tr( x1 + x2 )
                   

          x0 = self.conv01( xf )
          x0 = self.conv02( F.relu(x0) )
          x0 = torch.cat((self.conv021(x0), self.conv022(x0) * self.conv023(x0)),dim=1)
          x0 = self.conv03( x0 )
           
          x   = x1 + x0
          
          x = x.view(-1,shapeData[0],shapeData[1],1)
          
          # forecasting component
          x_forecast = self.ode_l63( x[:,0:3,shapeData[1]-self.dt_forecast-1,:] , self.dt_forecast )
          x_forecast = x_forecast.view(-1,3,self.dt_forecast,1)
          
          # concatenation
          xpred = 1. * x
          xpred[:,0:3,dT-dt_forecast:,:] = 1. * x_forecast
          
          return xpred

elif flagAEType == 'unet2+wc_ode': ## Conv model with no use of the central point
  dW = 5

  class Odenet_l63(torch.nn.Module):
        def __init__(self):
              super(Odenet_l63, self).__init__()
              self.sigma = torch.nn.Parameter(torch.Tensor([np.random.randn()]))
              self.rho    = torch.nn.Parameter(torch.Tensor([np.random.randn()]))
              self.beta   = torch.nn.Parameter(torch.Tensor([np.random.randn()]))

              self.sigma  = torch.nn.Parameter(torch.Tensor([10.]))
              self.rho    = torch.nn.Parameter(torch.Tensor([28.]))
              self.beta   = torch.nn.Parameter(torch.Tensor([8./3.]))

              self.dt        = 0.01
              self.IntScheme = 'rk4'
              self.stdTr     = stdTr
              self.meanTr    = meanTr                      
        def _odeL63(self, xin):
            x1  = xin[:,0,:]
            x2  = xin[:,1,:]
            x3  = xin[:,2,:]
            
            dx_1 = (self.sigma*(x2-x1)).view(-1,1,xin.size(2))
            dx_2 = (x1*(self.rho-x3)-x2).view(-1,1,xin.size(2))
            dx_3 = (x1*x2 - self.beta*x3).view(-1,1,xin.size(2))
            
            dpred = torch.cat((dx_1,dx_2,dx_3),dim=1)
            return dpred

        def _EulerSolver(self, x):
            return x + self.dt * self._odeL63(x)

        def _RK4Solver(self, x):
            k1 = self._odeL63(x)
            x2 = x + 0.5 * self.dt * k1
            k2 = self._odeL63(x2)
          
            x3 = x + 0.5 * self.dt * k2
            k3 = self._odeL63(x3)
              
            x4 = x + self.dt * k3
            k4 = self._odeL63(x4)

            return x + self.dt * (k1+2.*k2+2.*k3+k4)/6.
      
        def forward(self, x):
            X = self.stdTr * x.view(-1,x.size(1),x.size(2))
            X = X + self.meanTr
            
            if self.IntScheme == 'euler':
                xpred = self._EulerSolver( X[:,:,0:x.size(2)-1] )
            else:
                xpred = self._RK4Solver( X[:,:,0:x.size(2)-1] )

            xpred = xpred - self.meanTr
            xpred = xpred / self.stdTr

            xnew  = torch.cat((x[:,:,0].view(-1,x.size(1),1),xpred),dim=2)
            
            xnew = xnew.view(-1,x.size(1),x.size(2),1)
            
            return xnew

  class Phi_r(torch.nn.Module):
      def __init__(self):
          super(Phi_r, self).__init__()
          self.pool1  = torch.nn.AvgPool2d((5,1))
          self.pool2  = torch.nn.AvgPool2d((2,1))
          #self.conv1  = ConstrainedConv2d(shapeData[0],2*shapeData[0]*DimAE,(2*dW+1,1),padding=(dW,0),bias=False)

          self.conv0  = torch.nn.Conv2d(shapeData[0],shapeData[0]*DimAE,(2*dW+1,1),padding=(dW,0),bias=False)

          self.conv21  = torch.nn.Conv2d(shapeData[0]*DimAE,2*shapeData[0]*DimAE,(2*dW+1,1),padding=(dW,0),bias=False)
          self.conv22  = torch.nn.Conv2d(2*shapeData[0]*DimAE,shapeData[0]*DimAE,1,padding=0,bias=False)          
          self.conv221 = torch.nn.Conv2d(shapeData[0]*DimAE,shapeData[0]*DimAE,1,padding=0,bias=False)
          self.conv222 = torch.nn.Conv2d(shapeData[0]*DimAE,shapeData[0]*DimAE,1,padding=0,bias=False)
          self.conv223 = torch.nn.Conv2d(shapeData[0]*DimAE,shapeData[0]*DimAE,1,padding=0,bias=False)
          self.conv23  = torch.nn.Conv2d(2*shapeData[0]*DimAE,shapeData[0]*DimAE,1,padding=0,bias=False)
          self.conv2Tr = torch.nn.ConvTranspose2d(shapeData[0]*DimAE,shapeData[0]*DimAE,(2,1),stride=(2,1),bias=False)          

          self.conv11  = torch.nn.Conv2d(shapeData[0]*DimAE,2*shapeData[0]*DimAE,(2*dW+1,1),padding=(dW,0),bias=False)
          self.conv12  = torch.nn.Conv2d(2*shapeData[0]*DimAE,shapeData[0]*DimAE,1,padding=0,bias=False)          
          self.conv121 = torch.nn.Conv2d(shapeData[0]*DimAE,shapeData[0]*DimAE,1,padding=0,bias=False)
          self.conv122 = torch.nn.Conv2d(shapeData[0]*DimAE,shapeData[0]*DimAE,1,padding=0,bias=False)
          self.conv123 = torch.nn.Conv2d(shapeData[0]*DimAE,shapeData[0]*DimAE,1,padding=0,bias=False)
          self.conv13  = torch.nn.Conv2d(2*shapeData[0]*DimAE,shapeData[0]*DimAE,1,padding=0,bias=False)
          self.conv1Tr = torch.nn.ConvTranspose2d(shapeData[0]*DimAE,shapeData[0],(5,1),stride=(5,1),bias=False)          

          self.conv01  = torch.nn.Conv2d(shapeData[0]*DimAE,2*shapeData[0]*DimAE,(2*dW+1,1),padding=(dW,0),bias=False)
          self.conv02  = torch.nn.Conv2d(2*shapeData[0]*DimAE,shapeData[0]*DimAE,1,padding=0,bias=False)          
          self.conv021 = torch.nn.Conv2d(shapeData[0]*DimAE,shapeData[0]*DimAE,1,padding=0,bias=False)
          self.conv022 = torch.nn.Conv2d(shapeData[0]*DimAE,shapeData[0]*DimAE,1,padding=0,bias=False)
          self.conv023 = torch.nn.Conv2d(shapeData[0]*DimAE,shapeData[0]*DimAE,1,padding=0,bias=False)
          self.conv03  = torch.nn.Conv2d(2*shapeData[0]*DimAE,shapeData[0],1,padding=0,bias=False)

          self.ode_l63 = Odenet_l63()
          self.dt_forecast = dt_forecast          

      def forward(self, xinp):
          #x = self.fc1( torch.nn.Flatten(x) )
          #x = self.pool1( xinp )
          
          # assimilation component
          xf = self.conv0( xinp )
          
          x2 = self.pool2( self.pool1( xf ) )
          x2 = self.conv21( x2 )
          x2 = self.conv22( F.relu(x2) )
          x2 = torch.cat((self.conv221(x2), self.conv222(x2) * self.conv223(x2)),dim=1)
          x2 = self.conv23( x2 )
          x2 = self.conv2Tr( x2 )

          x1 = self.pool1( xf )
          x1 = self.conv11( x1 )
          x1 = self.conv12( F.relu(x1) )
          x1 = torch.cat((self.conv121(x1), self.conv122(x1) * self.conv123(x1)),dim=1)
          x1 = self.conv13( x1 )
          x1 = self.conv1Tr( x1 + x2 )
                   

          x0 = self.conv01( xf )
          x0 = self.conv02( F.relu(x0) )
          x0 = torch.cat((self.conv021(x0), self.conv022(x0) * self.conv023(x0)),dim=1)
          x0 = self.conv03( x0 )
           
          x   = x1 + x0
          
          x = x.view(-1,shapeData[0],shapeData[1],1)
          
          # forecasting component
          x_forecast = self.ode_l63( x[:,0:3,shapeData[1]-self.dt_forecast-1:,:] )
          x_forecast = x_forecast.view(-1,3,self.dt_forecast+1,1)
          
          # concatenation
          xpred = 1. * x
          xpred[:,0:3,dT-dt_forecast:,:] = 1. * x_forecast[:,0:3,1:,:]
          xpred[:,3:,dT-dt_forecast:,:] = 0. * xpred[:,3:,dT-dt_forecast:,:]
          
          return xpred
      
phi_r           = Phi_r()
print(' AE Model/Dynamical prior: '+flagAEType)
print(phi_r)
print('AE/Prior: Number of trainable parameters = %d'%(sum(p.numel() for p in phi_r.parameters() if p.requires_grad)))


class Model_H(torch.nn.Module):
    def __init__(self):
        super(Model_H, self).__init__()
        #self.DimObs = 1
        #self.dimObsChannel = np.array([shapeData[0]])
        self.dim_obs = 1
        self.dim_obs_channel = np.array([shapeData[0]])

        self.DimObs = 1
        self.dimObsChannel = np.array([shapeData[0]])

    def forward(self, x, y, mask):
        dyout = (x - y) * mask
        return dyout

class HParam:
    def __init__(self):
        self.iter_update     = []
        self.nb_grad_update  = []
        self.lr_update       = []
        self.n_grad          = 1
        self.dim_grad_solver = 10
        self.dropout         = 0.25
        self.w_loss          = []
        self.automatic_optimization = True

        self.alpha_proj    = 0.5
        self.alpha_mse_rec = 1.
        self.alpha_mse_for = 10.

        self.k_batch = 1


EPS_NORM_GRAD = 0. * 1.e-20  
import pytorch_lightning as pl

class LitModel(pl.LightningModule):
    def __init__(self,conf=HParam(),*args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        # hyperparameters
        self.hparams.iter_update     = [0, 20, 50, 70, 100, 150, 800]  # [0,2,4,6,9,15]
        self.hparams.nb_grad_update  = [5, 5, 10, 10, 15, 15, 20, 20, 20]  # [0,0,1,2,3,3]#[0,2,2,4,5,5]#
        self.hparams.lr_update       = [1e-3, 1e-4, 1e-4, 1e-5, 1e-4, 1e-5, 1e-5, 1e-6, 1e-7]
        
        self.hparams.n_grad          = self.hparams.nb_grad_update[0]
        self.hparams.k_n_grad        = 1
        self.hparams.dim_grad_solver = dimGradSolver
        self.hparams.dropout         = rateDropout
        self.hparams.dim_aug_state = dim_aug_state
        
        self.hparams.k_batch         = 1
        
        self.hparams.alpha_prior   = 0.5
        self.hparams.alpha_mse     = 1.#10*0.75#1.e0
        self.hparams.alpha_mse_rec = 10.#10*0.75#1.e0
        self.hparams.alpha_mse_for = 0.#*0.25#1.e1
        
        self.hparams.w_loss          = torch.nn.Parameter(torch.Tensor(w_loss), requires_grad=False)
        self.hparams.automatic_optimization = False#True#

        # main model
        self.model        = solver_4DVarNet.Solver_Grad_4DVarNN(Phi_r(), 
                                                            Model_H(), 
                                                            solver_4DVarNet.model_GradUpdateLSTM(shapeData, UsePriodicBoundary, self.hparams.dim_grad_solver, self.hparams.dropout, padding_mode='zeros'), 
                                                            None, None, shapeData, self.hparams.n_grad, EPS_NORM_GRAD)#, self.hparams.eps_norm_grad)
        self.w_loss       = self.hparams.w_loss # duplicate for automatic upload to gpu
        self.x_rec    = None # variable to store output of test method
        self.x_rec_obs = None
        self.curr = 0

        self.automatic_optimization = self.hparams.automatic_optimization
                

    def forward(self):
        return 1

    def configure_optimizers(self):
        optimizer   = optim.Adam([{'params': self.model.model_Grad.parameters(), 'lr': self.hparams.lr_update[0]},
                                      {'params': self.model.model_VarCost.parameters(), 'lr': self.hparams.lr_update[0]},
                                    {'params': self.model.phi_r.parameters(), 'lr': 0.5*self.hparams.lr_update[0]},
                                    ], lr=0.)
        return optimizer
    
    def on_epoch_start(self):
        # enfore acnd check some hyperparameters 
        #self.model.n_grad   = self.hparams.k_n_grad * self.hparams.n_grad 
        self.model.n_grad   = self.hparams.n_grad 
        
    def on_train_epoch_start(self):
        self.model.n_grad   = self.hparams.n_grad 

        opt = self.optimizers()
        if (self.current_epoch in self.hparams.iter_update) & (self.current_epoch > 0):
            indx             = self.hparams.iter_update.index(self.current_epoch)
            print('... Update Iterations number/learning rate #%d: NGrad = %d -- lr = %f'%(self.current_epoch,self.hparams.nb_grad_update[indx],self.hparams.lr_update[indx]))
            
            self.hparams.n_grad = self.hparams.nb_grad_update[indx]
            self.model.n_grad   = self.hparams.n_grad 
            
            mm = 0
            lrCurrent = self.hparams.lr_update[indx]
            lr = np.array([lrCurrent,lrCurrent,0.5*lrCurrent,0.])            
            for pg in opt.param_groups:
                pg['lr'] = lr[mm]# * self.hparams.learning_rate
                mm += 1
        
        #if self.current_epoch == 0 :     
        #    self.save_hyperparameters()
        # update training data loaders
        if 0*0: #self.current_epoch > 0 : #self.current_epoch % 1 == 0:
            training_dataset     = torch.utils.data.TensorDataset(torch.Tensor(x_train_Init[:idx_val:,:,:,:]),torch.Tensor(x_train_obs[:idx_val:,:,:,:]),torch.Tensor(mask_train[:idx_val:,:,:,:]),torch.Tensor(x_train[:idx_val:,:,:,:])) # create your datset
            val_dataset         = torch.utils.data.TensorDataset(torch.Tensor(x_train_Init[idx_val::,:,:,:]),torch.Tensor(x_train_obs[idx_val::,:,:,:]),torch.Tensor(mask_train[idx_val::,:,:,:]),torch.Tensor(x_train[idx_val::,:,:,:])) # create your datset
            test_dataset         = torch.utils.data.TensorDataset(torch.Tensor(x_test_Init),torch.Tensor(x_test_obs),torch.Tensor(mask_test),torch.Tensor(x_test)) # create your datset
            
            dataloaders = {
                'train': torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),
                'val': torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),
                'test': torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),
            }            

            # run current model
            print('.. Update dataset (x_init for training and validation dataset)')
            trainer.test(mod, test_dataloaders=dataloaders['train'])
            self.x_rec_training =  ( self.x_rec - meanTr ) /stdTr
            self.x_rec_training = self.x_rec_training.reshape(-1,self.x_rec_training.shape[1],self.x_rec_training.shape[2],1)
            print('1111')
            print(self.x_rec_training.shape )
            
            x_train_Init_new = x_train_Init
            print(x_train_Init_new.shape)
            
            idx_rand = np.random.binomial(1,0.1,(x_train_Init[:idx_val,:,:,:].shape[0],1,1,1))
            idx_rand = np.tile( idx_rand , (1,x_train_Init.shape[1],x_train_Init.shape[2],x_train_Init.shape[3]) )
            x_train_Init_new[:idx_val,:,:,:] = x_train_Init[:idx_val,:,:,:] * idx_rand + (1. - idx_rand ) * self.x_rec_training

            trainer.test(mod, test_dataloaders=dataloaders['val'])
            self.x_rec_val =  ( self.x_rec - meanTr ) /stdTr
            self.x_rec_val = self.x_rec_val.reshape(-1,self.x_rec_val.shape[1],self.x_rec_val.shape[2],1)
            print( self.x_rec_val.shape )

            idx_rand = np.random.binomial(1,0.1,(x_train_Init[idx_val:,:,:,:].shape[0],1,1,1))
            idx_rand = np.tile( idx_rand , (1,x_train_Init.shape[1],x_train_Init.shape[2],x_train_Init.shape[3]) )
            x_train_Init_new[idx_val:,:,:,:] = x_train_Init[idx_val:,:,:,:] * idx_rand + (1. - idx_rand ) * self.x_rec_val
            print('2222')
            print(x_train_Init_new.shape)

            # update dataloader            
            training_dataset_new     = torch.utils.data.TensorDataset(torch.Tensor(x_train_Init_new[:idx_val:,:,:,:]),torch.Tensor(x_train_obs[:idx_val:,:,:,:]),torch.Tensor(mask_train[:idx_val:,:,:,:]),torch.Tensor(x_train[:idx_val:,:,:,:])) # create your datset
            val_dataset         = torch.utils.data.TensorDataset(torch.Tensor(x_train_Init[idx_val::,:,:,:]),torch.Tensor(x_train_obs[idx_val::,:,:,:]),torch.Tensor(mask_train[idx_val::,:,:,:]),torch.Tensor(x_train[idx_val::,:,:,:])) # create your datset
            
            dataloaders = {
                'train': torch.utils.data.DataLoader(training_dataset_new, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
                'val': torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),
                'test': torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),
            }            
        
    def training_step(self, train_batch, batch_idx, optimizer_idx=0):
        opt = self.optimizers()
                    
        # compute loss and metrics
        loss, out, metrics = self.compute_loss(train_batch, phase='train')
        
        for kk in range(0,self.hparams.k_n_grad-1):
            loss1, out, metrics = self.compute_loss(train_batch, phase='train',batch_init=out[0],hidden=out[1],cell=out[2],normgrad=out[3])
            loss = loss + loss1
        
        # log step metric        
        #self.log('train_mse', mse)
        #self.log("dev_loss", mse / var_Tr , on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        #self.log("loss", loss , on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("tr_mse", stdTr**2 * metrics['mse'] , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # initial grad value
        if self.hparams.automatic_optimization == False :
            # backward
            self.manual_backward(loss)
        
            if (batch_idx + 1) % self.hparams.k_batch == 0:
                # optimisation step
                opt.step()
                
                # grad initialization to zero
                opt.zero_grad()
         
        return {"training_loss": loss,'preds':out[0].detach().cpu(),'idx':out[4].detach().cpu()}
    
    def validation_step(self, val_batch, batch_idx):
        loss, out, metrics = self.compute_loss(val_batch, phase='val')
        for kk in range(0,self.hparams.k_n_grad-1):
            loss1, out, metrics = self.compute_loss(val_batch, phase='val',batch_init=out[0],hidden=out[1],cell=out[2],normgrad=out[3])
            loss = loss1

        #self.log('val_loss', loss)
        self.log('val_loss', stdTr**2 * metrics['mse'] )
        self.log("val_mse", stdTr**2 * metrics['mse'] , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        #return self.log('val_loss', loss)
        return {"val_loss": loss,'preds':out[0].detach().cpu()}

    def test_step(self, test_batch, batch_idx):
        loss, out, metrics = self.compute_loss(test_batch, phase='test')
        
        for kk in range(0,self.hparams.k_n_grad-1):
            loss1, out, metrics = self.compute_loss(test_batch, phase='test',batch_init=out[0].detach(),hidden=out[1],cell=out[2],normgrad=out[3])

        #out_ssh,out_ssh_obs = out
        #self.log('test_loss', loss)
        self.log("test_mse", stdTr**2 * metrics['mse'] , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        #return {'preds': out_ssh.detach().cpu(),'obs_ssh': out_ssh_obs.detach().cpu()}
        return {'preds': out[0].detach().cpu()}

#    def training_epoch_end(self, training_step_outputs):
#        # do something with all training_step outputs
        #print('.. \n')
    
    def training_epoch_end(self, outputs):
        x_rec_curr = torch.cat([chunk['preds'] for chunk in outputs]).numpy()
        idx_rec_curr = torch.cat([chunk['idx'] for chunk in outputs]).numpy()
        idx_rec_curr = idx_rec_curr.astype(int)
        
        self.x_rec_training = x_rec_curr[idx_rec_curr,:,:,:]
                
        loss_val = torch.stack([x['training_loss'] for x in outputs]).mean()
        self.log('train_loss_epoch', loss_val)

    def validation_epoch_end(self, outputs):
        x_rec_curr = torch.cat([chunk['preds'] for chunk in outputs]).numpy()
        self.x_rec_val = x_rec_curr
                
        loss_val = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val_loss_epoch', loss_val)
        
    def test_epoch_end(self, outputs):
        x_test_rec = torch.cat([chunk['preds'] for chunk in outputs]).numpy()
        
        if self.hparams.dim_aug_state > 0 :
            x_test_rec = x_test_rec[:,:3,:]

        x_test_rec = stdTr * x_test_rec + meanTr        
        self.x_rec = x_test_rec.squeeze()

        return [{'mse':0.,'preds': 0.}]

    def compute_loss(self, batch, phase, batch_init = None , hidden = None , cell = None , normgrad = 0.0):

        idx,inputs_init_,inputs_obs,masks,targets_GT = batch
 
        #inputs_init = inputs_init_
        if batch_init is None :
            if self.current_epoch > 0:
                idx_init = idx.cpu().numpy().astype(int)
                ind0 = np.random.permutation(inputs_init_.size(0))
                n0 = int( 0.2 * inputs_init_.size(0) )
                
                ind0 = ind0[:n0]
                ind0_init = idx_init[ ind0 ]
                
                if phase == 'train' :       
                    inputs_init_[ind0,:,:,:] = torch.Tensor(self.x_rec_training[ind0_init,:3,:,:]).to(device)
                    
            if self.hparams.dim_aug_state == 0 :   
                inputs_init = inputs_init_
            else:                
                #init_aug_state = 0. * inputs_init_[:,0,:,:]
                #init_aug_state = init_aug_state.view(-1,1,inputs_init_.size(2),1)
                #init_aug_state = init_aug_state.repeat(1,dim_aug_state,1,1)
                
                init_aug_state = 0.1 * torch.randn((inputs_init_.size(0),self.hparams.dim_aug_state,inputs_init_.size(2),inputs_init_.size(3)))
                inputs_init = torch.cat( (inputs_init_,init_aug_state.to(device)) , dim = 1 )
        else:
            inputs_init = batch_init
        
        if self.hparams.dim_aug_state > 0 :   
            
            init_aug_state = 0. * inputs_init_[:,0,:,:]
            init_aug_state = init_aug_state.view(-1,1,inputs_init_.size(2),1)
            init_aug_state = init_aug_state.repeat(1,dim_aug_state,1,1)
            masks = torch.cat( (masks,init_aug_state) , dim = 1 )
            inputs_obs = torch.cat( (inputs_obs,init_aug_state) , dim = 1 )
                        
        if phase == 'train' :                
            inputs_init = inputs_init.detach()
            
        with torch.set_grad_enabled(True):
            # with torch.set_grad_enabled(phase == 'train'):
            inputs_init = torch.autograd.Variable(inputs_init, requires_grad=True)

            #outputs, hidden_new, cell_new, normgrad_ = self.model(inputs_init_, inputs_obs, masks)#,hidden = hidden , cell = cell , normgrad = normgrad)
            #outputs, hidden_new, cell_new, normgrad_ = self.model(inputs_init, inputs_obs, masks ,hidden = None , cell = None , normgrad = normgrad )
            outputs, hidden_new, cell_new, normgrad_ = self.model(inputs_init, inputs_obs, masks, hidden = hidden , cell = cell , normgrad = normgrad )

            if self.hparams.dim_aug_state == 0 : 
                if flag_x1_only == False:
                    loss_mse = torch.mean((outputs - targets_GT) ** 2)
                    loss_mse_rec = torch.mean((outputs[:,:,:dT-dt_forecast,:] - targets_GT[:,:,:dT-dt_forecast,:]) ** 2)
                    loss_mse_for = torch.mean((outputs[:,:,dT-dt_forecast:,:] - targets_GT[:,:,dT-dt_forecast:,:]) ** 2)
                else:
                    loss_mse_rec = torch.mean((outputs[:,0,:dT-dt_forecast,:] - targets_GT[:,0,:dT-dt_forecast,:]) ** 2)
                    loss_mse_for = torch.mean((outputs[:,0,dT-dt_forecast:,:] - targets_GT[:,0,dT-dt_forecast:,:]) ** 2)
                    
                loss_prior = torch.mean((self.model.phi_r(outputs) - outputs) ** 2)
                
                loss_prior_gt = torch.mean((self.model.phi_r(targets_GT) - targets_GT) ** 2)
            else:
                if flag_x1_only == False:
                    loss_mse = torch.mean((outputs[:,:3,:,:] - targets_GT[:,:,:,:]) ** 2)
                    loss_mse_rec = torch.mean((outputs[:,:3,:dT-dt_forecast,:] - targets_GT[:,:,:dT-dt_forecast,:]) ** 2)
                    loss_mse_for = torch.mean((outputs[:,:3,dT-dt_forecast:,:] - targets_GT[:,:,dT-dt_forecast:,:]) ** 2)
                else:
                    loss_mse_rec = torch.mean((outputs[:,0,:dT-dt_forecast,:] - targets_GT[:,0,:dT-dt_forecast,:]) ** 2)
                    loss_mse_for = torch.mean((outputs[:,0,dT-dt_forecast:,:] - targets_GT[:,0,dT-dt_forecast:,:]) ** 2)

                loss_prior = torch.mean((self.model.phi_r(outputs) - outputs) ** 2)
                
                targets_gt_aug = torch.cat( (targets_GT,outputs[:,3:,:]) , dim= 1)
                loss_prior_gt = torch.mean((self.model.phi_r(targets_gt_aug) - targets_gt_aug) ** 2)
                
            #loss_mse   = solver_4DVarNet.compute_WeightedLoss((outputs - targets_GT), self.w_loss)

            if flagForecast == True :
                loss_mse = self.hparams.alpha_mse_rec * loss_mse_rec + self.hparams.alpha_mse_for * loss_mse_for
            else:
                loss_mse = self.hparams.alpha_mse * loss_mse
            loss = loss_mse + 0.5 * self.hparams.alpha_prior * (loss_prior + loss_prior_gt)
            
            # metrics
            mse       = loss_mse.detach()
            metrics   = dict([('mse',mse)])
            #print(mse.cpu().detach().numpy())
            if (phase == 'val') or (phase == 'test'):                
                outputs = outputs.detach()
        
        out = [outputs,hidden_new, cell_new, normgrad_,idx]
        
        return loss,out, metrics


class Ode_l63(torch.nn.Module):
    def __init__(self):
          super(Ode_l63, self).__init__()
          self.sigma = torch.nn.Parameter(torch.Tensor([np.random.randn()]))
          self.rho    = torch.nn.Parameter(torch.Tensor([np.random.randn()]))
          self.beta   = torch.nn.Parameter(torch.Tensor([np.random.randn()]))

          self.sigma  = torch.nn.Parameter(torch.Tensor([10.]))
          self.rho    = torch.nn.Parameter(torch.Tensor([28.]))
          self.beta   = torch.nn.Parameter(torch.Tensor([8./3.]))

          self.dt        = 0.01
          self.IntScheme = 'rk4'
          self.stdTr     = stdTr
          self.meanTr    = meanTr                      
    def _odeL63(self, xin):
        x1  = xin[:,0,:]
        x2  = xin[:,1,:]
        x3  = xin[:,2,:]
        
        dx_1 = (self.sigma*(x2-x1)).view(-1,1,xin.size(2))
        dx_2 = (x1*(self.rho-x3)-x2).view(-1,1,xin.size(2))
        dx_3 = (x1*x2 - self.beta*x3).view(-1,1,xin.size(2))
        
        dpred = torch.cat((dx_1,dx_2,dx_3),dim=1)
        return dpred

    def _EulerSolver(self, x):
        return x + self.dt * self._odeL63(x)

    def _RK4Solver(self, x):
        k1 = self._odeL63(x)
        x2 = x + 0.5 * self.dt * k1
        k2 = self._odeL63(x2)
      
        x3 = x + 0.5 * self.dt * k2
        k3 = self._odeL63(x3)
          
        x4 = x + self.dt * k3
        k4 = self._odeL63(x4)

        return x + self.dt * (k1+2.*k2+2.*k3+k4)/6.
  
    def ode_int(self, x0, N):
        X = self.stdTr * x0.view(-1,x0.size(1),1) + self.meanTr
        
        out = 1. * X
        
        for nn in range(0,N):
            X = self._RK4Solver( X )
            out = torch.cat( (out,X) , dim = 2 )
     
        out = out - self.meanTr
        out = out / self.stdTr
            
        return out
     
    def forward(self, x):
        X = self.stdTr * x.view(-1,x.size(1),x.size(2))
        X = X + self.meanTr
        
        if self.IntScheme == 'euler':
            xpred = self._EulerSolver( X[:,:,0:x.size(2)-1] )
        else:
            xpred = self._RK4Solver( X[:,:,0:x.size(2)-1] )

        xpred = xpred - self.meanTr
        xpred = xpred / self.stdTr

        xnew  = torch.cat((x[:,:,0].view(-1,x.size(1),1),xpred),dim=2)
        
        xnew = xnew.view(-1,x.size(1),x.size(2),1)
        
        return xnew


class LitModel_4dvar_classic(pl.LightningModule):
    def __init__(self,conf=HParam(),*args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        # main model
        self.phi = Ode_l63() 
        self.lam_grad = 0.1
        self.alpha_obs = 1.
        self.alpha_prior = 1.
        self.n_iter_descent = 100
        self.dt_forecast = dt_forecast
        
        self.x_rec    = None # variable to store output of test method
        self.flag_ode_forecast = False#True
        self.x_rec_training = None
        
    def forward(self):
        return 1

    def configure_optimizers(self):
        optimizer   = optim.Adam([{'params': self.model.model_Grad.phi(), 'lr': 0.}], lr=0.)
        return optimizer
            
    def on_train_epoch_start(self):

        print('...  No training step for classic 4DVar method')
        
        #if self.ncurret_epoch == 0 :     
        #    self.save_hyperparameters()
        
    def training_step(self, train_batch, batch_idx, optimizer_idx=0):
                     
        return 0.
    
    def validation_step(self, val_batch, batch_idx):
        loss, out, metrics = self.compute_loss(val_batch, phase='val')

        #self.log('val_loss', loss)
        self.log('val_loss', stdTr**2 * metrics['mse'] )
        self.log("val_mse", stdTr**2 * metrics['mse'] , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, test_batch, batch_idx):
        loss, out, metrics = self.compute_loss(test_batch, phase='test')
        
        #out_ssh,out_ssh_obs = out
        #self.log('test_loss', loss)
        self.log("test_mse", stdTr**2 * metrics['mse'] , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        #return {'preds': out_ssh.detach().cpu(),'obs_ssh': out_ssh_obs.detach().cpu()}
        return {'preds': out[0].detach().cpu()}

#    def training_epoch_end(self, training_step_outputs):
#        # do something with all training_step outputs
#        #print('.. \n')

    
    def test_epoch_end(self, outputs):
        x_test_rec = torch.cat([chunk['preds'] for chunk in outputs]).numpy()

        x_test_rec = stdTr * x_test_rec + meanTr        
        self.x_rec = x_test_rec.squeeze()
        print( self.x_rec.shape )
    
        return [{'mse':0.,'preds': 0.}]

    def compute_loss(self, batch, phase, batch_init = None , hidden = None , cell = None , normgrad = 0.0):

        inputs_init_,inputs_obs,masks,targets_GT = batch
 
        #inputs_init = inputs_init_
        if batch_init is None :
            inputs_init = inputs_init_
        else:
            inputs_init = batch_init
        
            
        with torch.set_grad_enabled(True):
            # with torch.set_grad_enabled(phase == 'train'):
            x_curr = torch.autograd.Variable(inputs_init, requires_grad=True)

            for iter in range(0,self.n_iter_descent):
                # prior term
                if self.flag_ode_forecast == True :
                    loss_prior = torch.mean( (x_curr[:,:,:dT-dt_forecast] - self.phi(x_curr[:,:,:dT-dt_forecast] ))**2  )
                    loss_obs = torch.mean( (x_curr[:,:,:dT-dt_forecast] - inputs_obs[:,:,:dT-dt_forecast] )**2 * masks[:,:,:dT-dt_forecast] )
                else:
                    loss_prior = torch.mean( (x_curr - self.phi(x_curr ))**2  )
                    loss_obs = torch.mean( (x_curr - inputs_obs )**2 * masks )
                
                # overall loss
                loss = self.alpha_obs * loss_obs + self.alpha_prior * loss_prior 

                if( np.mod(iter,100) == 0 ):
                    if self.flag_ode_forecast == True :
                        mse = torch.mean( (x_curr[:,:,:dT-dt_forecast-1,:] - targets_GT[:,:,:dT-dt_forecast-1,:] )**2  )
                    else:
                        mse = torch.mean( (x_curr - targets_GT )**2  )

                    print(".... iter %d: loss %.3f dyn_loss %.3f obs_loss %.3f mse %.3f"%(iter,1.e3*loss,1.e3*loss_prior,1.e3*loss_obs,stdTr**2 * mse))  

                # compute gradient w.r.t. X and update X
                loss.backward()
                #print( torch.sqrt( torch.mean(  x_curr.grad.data ** 2 ) ))
                x_curr = x_curr - self.lam * x_curr.grad.data
                x_curr = torch.autograd.Variable(x_curr, requires_grad=True)

            outputs = 1. * x_curr
            
            if self.flag_ode_forecast == True :
                x_for = self.phi.ode_int( x_curr[:,:,dT-dt_forecast-1,:] , dt_forecast )
                
                outputs[:,:,dT-dt_forecast-1:] = x_for.view(-1,3,dt_forecast+1,1)
                    
            loss_mse = torch.mean((outputs - targets_GT) ** 2)
                                         
            # metrics
            mse       = loss_mse.detach()
            metrics   = dict([('mse',mse)])
            #print(mse.cpu().detach().numpy())
            if (phase == 'val') or (phase == 'test'):                
                outputs = outputs.detach()
        
        out = [outputs]
        
        return loss,out, metrics

class HParam_FixedPoint:
    def __init__(self):
        self.n_iter_fp       = 1
        self.k_n_fp       = 1

        self.alpha_proj    = 0.5
        self.alpha_mse = 1.
        self.lr = 1.e-3

EPS_NORM_GRAD = 0. * 1.e-20  
import pytorch_lightning as pl

    
from pytorch_lightning.callbacks import ModelCheckpoint

UsePriodicBoundary = False # use a periodic boundary for all conv operators in the gradient model (see torch_4DVarNN_dinAE)
if flagForecast == True :
    w_loss = np.ones(dT) / np.float(dT)
    w_loss[dT-dt_forecast:] = 1. / np.float(dt_forecast)
else:
    w_loss = np.ones(dT) / np.float(dT)


if __name__ == '__main__':
      
    if flagProcess == 0: ## training model from scratch
        
        flagLoadModel = False#True #
        if flagLoadModel == True:
            
            pathCheckPOint = 'resL63/exp02-2/model-l63-forecast_055-aug10-unet2-exp02-2-Noise01-igrad05_02-dgrad25-drop20-epoch=105-val_loss=2.08.ckpt'
            pathCheckPOint = 'resL63/exp02-2/model-l63-aug10-unet2-exp02-2-Noise01-igrad05_02-dgrad25-drop20-epoch=117-val_loss=0.55.ckpt'
            
            print('.... load pre-trained model :'+pathCheckPOint)
            mod = LitModel.load_from_checkpoint(pathCheckPOint)

            mod.hparams.n_grad          = 5
            mod.hparams.k_n_grad        = 4
            mod.hparams.iter_update     = [0, 100, 200, 300, 500, 700, 800]  # [0,2,4,6,9,a15]
            mod.hparams.nb_grad_update  = [10, 10, 10, 10, 10, 5, 20, 20, 20]  # [0,0,1,2,3,3]#[0,2,2,4,5,5]#
            mod.hparams.lr_update       = [1e-4, 1e-5, 1e-6, 1e-5, 1e-4, 1e-5, 1e-5, 1e-6, 1e-7]
        else:
            mod = LitModel()
            
            mod.hparams.n_grad          = 5
            mod.hparams.k_n_grad        = 2
            mod.hparams.iter_update     = [0, 100, 200, 300, 500, 700, 800]  # [0,2,4,6,9,15]
            mod.hparams.nb_grad_update  = [5, 5, 10, 10, 15, 15, 20, 20, 20]  # [0,0,1,2,3,3]#[0,2,2,4,5,5]#
            mod.hparams.lr_update       = [1e-3, 1e-4, 1e-4, 1e-5, 1e-4, 1e-5, 1e-5, 1e-6, 1e-7]
        
        mod.hparams.alpha_prior = 0.1
        mod.hparams.alpha_mse = 1.0#0.75
        mod.hparams.alpha_mse_rec = (dT-dt_forecast)/dT #0.75
        mod.hparams.alpha_mse_for = dt_forecast/dT #0.5#0.25

        mod.x_rec_training = x_train_Init[:idx_val,:,:,:]
        mod.x_rec_val = x_train_Init[idx_val:,:,:,:]
        
        profiler_kwargs = {'max_epochs': 200 }

        suffix_exp = 'exp%02d-2-testloaders'%flagTypeMissData
        filename_chkpt = 'model-l63-'
        
        if flagForecast == True :
            filename_chkpt = filename_chkpt+'forecast_%03d-'%dt_forecast
        
        if mod.hparams.alpha_mse_rec == 0. :
            filename_chkpt = filename_chkpt+'-norec-'
            
        if flag_x1_only == True :
            filename_chkpt = filename_chkpt+'x1_only-'
            
        if dim_aug_state > 0 :
            filename_chkpt = filename_chkpt+'aug%02d-'%dim_aug_state
                  
        filename_chkpt = filename_chkpt+flagAEType+'-'  
            
        filename_chkpt = filename_chkpt + suffix_exp+'-Noise%02d'%(sigNoise)
        filename_chkpt = filename_chkpt+'-igrad%02d_%02d'%(mod.hparams.n_grad,mod.hparams.k_n_grad)+'-dgrad%d'%dimGradSolver
        filename_chkpt = filename_chkpt+'-drop%02d'%(100*rateDropout)

        print('.... chkpt: '+filename_chkpt)
        checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                              dirpath= './resL63/'+suffix_exp,
                                              filename= filename_chkpt + '-{epoch:02d}-{val_loss:.2f}',
                                              save_top_k=3,
                                              mode='min')
        trainer = pl.Trainer(gpus=1,  **profiler_kwargs,callbacks=[checkpoint_callback])
        trainer.fit(mod, dataloaders['train'], dataloaders['val'])
        
    elif flagProcess == 1: ## test trained model from pre-trained model

        pathCheckPOint = 'resL63/exp02-2/model-l63-aug10-unet2-exp02-2-Noise01-igrad05_02-dgrad25-drop20-epoch=117-val_loss=0.55.ckpt'
        #pathCheckPOint = 'resL63/exp02-2/model-l63-aug03-unet2-exp02-2-Noise01-igrad05_02-dgrad25-drop20-epoch=50-val_loss=0.71.ckpt'
        
        #pathCheckPOint = 'resL63/exp02-2/model-l63-forecast_103-aug10-unet2-exp02-2-Noise01-igrad05_02-dgrad50-drop20-epoch=33-val_loss=9.40.ckpt'
        pathCheckPOint = 'resL63/exp02-2/model-l63-forecast_055-aug10-unet2-exp02-2-Noise01-igrad05_02-dgrad25-drop20-epoch=105-val_loss=2.08.ckpt'
        print('.... load pre-trained model :'+pathCheckPOint)
        
        mod = LitModel.load_from_checkpoint(pathCheckPOint)            
        
        print(mod.hparams)
        mod.hparams.alpha_mse = 1.
        mod.hparams.alpha_mse_rec = 0.75
        mod.hparams.alpha_mse_for = 0.25
        mod.hparams.n_grad = 5
        mod.hparams.k_n_grad = 2
    
        print(' Ngrad = %d / %d'%(mod.hparams.n_grad,mod.model.n_grad))
        #trainer = pl.Trainer(gpus=1, accelerator = "ddp", **profiler_kwargs)

        profiler_kwargs = {'max_epochs': 1}
        trainer = pl.Trainer(gpus=1,  **profiler_kwargs)
        
        #trainer.fit(mod, dataloaders['train'], dataloaders['val'])
        
        if 1*1 :
            trainer.test(mod, test_dataloaders=dataloaders['val'])
            
            # Reconstruction performance
            X_val = X_train[idx_val::,:,:]
            mask_val = mask_train[idx_val::,:,:,:].squeeze()
            var_val  = np.mean( (X_val - np.mean(X_val,axis=0))**2 )
            mse = np.mean( (mod.x_rec-X_val) **2 ) 
            mse_i   = np.mean( (1.-mask_val.squeeze()) * (mod.x_rec-X_val) **2 ) / np.mean( (1.-mask_val) )
            mse_r   = np.mean( mask_val.squeeze() * (mod.x_rec-X_val) **2 ) / np.mean( mask_val )
            
            nmse = mse / var_val
            nmse_i = mse_i / var_val
            nmse_r = mse_r / var_val
            
            print("..... Assimilation performance (validation data)")
            print(".. MSE ALL.   : %.3f / %.3f"%(mse,nmse))
            print(".. MSE ObsData: %.3f / %.3f"%(mse_r,nmse_r))
            print(".. MSE Interp : %.3f / %.3f"%(mse_i,nmse_i))
        
        trainer.test(mod, test_dataloaders=dataloaders['test'])
        print(' Ngrad = %d / %d'%(mod.hparams.n_grad,mod.model.n_grad))

        # Reconstruction performance
        if flagForecast == True :
            var_test  = np.mean( (X_test - np.mean(X_test,axis=0))**2 )
            mse_rec = np.mean( (mod.x_rec[:,:,:dT-dt_forecast]-X_test[:,:,:dT-dt_forecast]) **2 ) 
            
            nmse_rec = mse_rec / var_test
            
            print("..... Assimilation performance (test data)")
            print(".. MSE rec   : %.3f / %.3f"%(mse_rec,nmse_rec))
            
            print("\n")
            print('..... Forecasting performance (all):')
            for nn in range(0,dt_forecast+1):
                var_test_nn     = np.mean( (X_test[:,:,dT-dt_forecast+nn-1] - X_test[:,:,dT-dt_forecast-1])**2 )
                mse_forecast = np.mean( (mod.x_rec[:,:,dT-dt_forecast+nn-1]-X_test[:,:,dT-dt_forecast+nn-1]) **2 ) 
            
                print('... dt [ %03d ] = %.3f / %.3f / %.3f '%(nn,mse_forecast,mse_forecast/var_test,mse_forecast/var_test_nn) )                
            
            print("\n")
            print('..... Forecasting performance (x1):')
            var_test_x1  = np.mean( (X_test[:,0,:] - np.mean(X_test[:,0,:],axis=0))**2 )
            for nn in range(0,dt_forecast+1):
                var_test_nn     = np.mean( (X_test[:,0,dT-dt_forecast+nn-1] - X_test[:,0,dT-dt_forecast-1])**2 )
                mse_forecast = np.mean( (mod.x_rec[:,0,dT-dt_forecast+nn-1]-X_test[:,0,dT-dt_forecast+nn-1]) **2 ) 
            
                print('... dt [ %03d ] = %.3f / %.3f / %.3f '%(nn,mse_forecast,mse_forecast/var_test,mse_forecast/var_test_nn) )                
        else:
            var_test  = np.mean( (X_test - np.mean(X_test,axis=0))**2 )
            mse = np.mean( (mod.x_rec-X_test) **2 ) 
            mse_i   = np.mean( (1.-mask_test.squeeze()) * (mod.x_rec-X_test) **2 ) / np.mean( (1.-mask_test) )
            mse_r   = np.mean( mask_test.squeeze() * (mod.x_rec-X_test) **2 ) / np.mean( mask_test )
            
            nmse = mse / var_test
            nmse_i = mse_i / var_test
            nmse_r = mse_r / var_test
            
            print("..... Assimilation performance (test data)")
            print(".. MSE ALL.   : %.3f / %.3f"%(mse,nmse))
            print(".. MSE ObsData: %.3f / %.3f"%(mse_r,nmse_r))
            print(".. MSE Interp : %.3f / %.3f"%(mse_i,nmse_i))     
                
        import xarray as xr
        xrdata = xr.Dataset( data_vars={'l63-rec': (["n", "D", "dT"],mod.x_rec),'l63-gt': (["n", "D", "dT"],X_test)})
        xrdata.to_netcdf(path=pathCheckPOint.replace('.ckpt','_res.nc'), mode='w')


    if flagProcess == 2: # WC 4DVar solution using a fixed-step gradient descent
        
        mod = LitModel_4dvar_classic()            
        
        mod.alpha_prior = 1e4
        mod.alpha_obs = 1e5
        mod.lam = 2e-3 * batch_size 
        mod.n_iter_descent = 22000
        mod.flag_ode_forecast = True#
    
        #trainer = pl.Trainer(gpus=1, accelerator = "ddp", **profiler_kwargs)

        profiler_kwargs = {'max_epochs': 1}
        trainer = pl.Trainer(gpus=1,  **profiler_kwargs)
        
        #trainer.fit(mod, dataloaders['train'], dataloaders['val'])
        
        if 1*0 :
            trainer.test(mod, test_dataloaders=dataloaders['val'])
            
            # Reconstruction performance
            X_val = X_train[idx_val::,:,:]
            mask_val = mask_train[idx_val::,:,:,:].squeeze()
            var_val  = np.mean( (X_val - np.mean(X_val,axis=0))**2 )
            mse = np.mean( (mod.x_rec-X_val) **2 ) 
            mse_i   = np.mean( (1.-mask_val.squeeze()) * (mod.x_rec-X_val) **2 ) / np.mean( (1.-mask_val) )
            mse_r   = np.mean( mask_val.squeeze() * (mod.x_rec-X_val) **2 ) / np.mean( mask_val )
            
            nmse = mse / var_val
            nmse_i = mse_i / var_val
            nmse_r = mse_r / var_val
            
            print("..... Assimilation performance (validation data)")
            print(".. MSE ALL.   : %.3f / %.3f"%(mse,nmse))
            print(".. MSE ObsData: %.3f / %.3f"%(mse_r,nmse_r))
            print(".. MSE Interp : %.3f / %.3f"%(mse_i,nmse_i))
        
        trainer.test(mod, test_dataloaders=dataloaders['test'])

        # Reconstruction performance
        if flagForecast == True :
            var_test  = np.mean( (X_test - np.mean(X_test,axis=0))**2 )
            mse_rec = np.mean( (mod.x_rec[:,:,:dT-dt_forecast]-X_test[:,:,:dT-dt_forecast]) **2 ) 
            
            nmse_rec = mse_rec / var_test
            
            print("..... Assimilation performance (test data)")
            print(".. MSE rec   : %.3f / %.3f"%(mse_rec,nmse_rec))
            
            print("\n")
            print('..... Forecasting performance (all):')
            for nn in range(0,dt_forecast+1):
                var_test_nn     = np.mean( (X_test[:,:,dT-dt_forecast+nn-1] - X_test[:,:,dT-dt_forecast-1])**2 )
                mse_forecast = np.mean( (mod.x_rec[:,:,dT-dt_forecast+nn-1]-X_test[:,:,dT-dt_forecast+nn-1]) **2 ) 
            
                print('... dt [ %03d ] = %.3f / %.3f / %.3f '%(nn,mse_forecast,mse_forecast/var_test,mse_forecast/var_test_nn) )                
            
            print("\n")
            print('..... Forecasting performance (x1):')
            var_test_x1  = np.mean( (X_test[:,0,:] - np.mean(X_test[:,0,:],axis=0))**2 )
            for nn in range(0,dt_forecast+1):
                var_test_nn     = np.mean( (X_test[:,0,dT-dt_forecast+nn-1] - X_test[:,0,dT-dt_forecast-1])**2 )
                mse_forecast = np.mean( (mod.x_rec[:,0,dT-dt_forecast+nn-1]-X_test[:,0,dT-dt_forecast+nn-1]) **2 ) 
            
                print('... dt [ %03d ] = %.3f / %.3f / %.3f '%(nn,mse_forecast,mse_forecast/var_test,mse_forecast/var_test_nn) )                
        else:
            var_test  = np.mean( (X_test - np.mean(X_test,axis=0))**2 )
            mse = np.mean( (mod.x_rec-X_test) **2 ) 
            mse_i   = np.mean( (1.-mask_test.squeeze()) * (mod.x_rec-X_test) **2 ) / np.mean( (1.-mask_test) )
            mse_r   = np.mean( mask_test.squeeze() * (mod.x_rec-X_test) **2 ) / np.mean( mask_test )
            
            nmse = mse / var_test
            nmse_i = mse_i / var_test
            nmse_r = mse_r / var_test
            
            print("..... Assimilation performance (test data)")
            print(".. MSE ALL.   : %.3f / %.3f"%(mse,nmse))
            print(".. MSE ObsData: %.3f / %.3f"%(mse_r,nmse_r))
            print(".. MSE Interp : %.3f / %.3f"%(mse_i,nmse_i))     
            
        if 1*1 :
            xrdata = xr.Dataset( data_vars={'l63-rec': (["n", "D", "dT"],mod.x_rec),'l63-gt': (["n", "D", "dT"],X_test)})
            xrdata.to_netcdf(path='/tmp/res_l63_4dvar_classic_res.nc', mode='w')
        
    elif flagProcess == 3: ## testing trainable fixed-point scheme
        dimGradSolver = 25
        rateDropout = 0.2

        pathCheckPOint = 'resL63/exp02-2/model-l63-unet-exp02-2-fp05_01-epoch=112-val_loss=1.66.ckpt'
        
        #pathCheckPOint = 'resL63/exp02-2/model-l63-ode-exp02-2-igrad05_02-dgrad25-drop_20-epoch=405-val_loss=5.89.ckpt'
        
        print('.... load pre-trained model :'+pathCheckPOint)
        
        mod = LitModel_FixedPoint.load_from_checkpoint(pathCheckPOint)            
            
        mod.hparams.n_iter_fp       = 5
        mod.hparams.k_n_fp          = 1
    
        print(' Nb projection iterations = %d / %d'%(mod.hparams.n_iter_fp,mod.hparams.k_n_fp))
        #trainer = pl.Trainer(gpus=1, accelerator = "ddp", **profiler_kwargs)

        profiler_kwargs = {'max_epochs': 1}
        trainer = pl.Trainer(gpus=1,  **profiler_kwargs)
        
        #trainer.fit(mod, dataloaders['train'], dataloaders['val'])
        
        trainer.test(mod, test_dataloaders=dataloaders['val'])
        
        # Reconstruction performance
        X_val = X_train[idx_val::,:,:]
        mask_val = mask_train[idx_val::,:,:,:].squeeze()
        var_val  = np.mean( (X_val - np.mean(X_val,axis=0))**2 )
        
        print(mod.x_rec.shape)
        print(X_val.shape)
        mse = np.mean( (mod.x_rec-X_val) **2 ) 
        mse_i   = np.mean( (1.-mask_val.squeeze()) * (mod.x_rec-X_val) **2 ) / np.mean( (1.-mask_val) )
        mse_r   = np.mean( mask_val.squeeze() * (mod.x_rec-X_val) **2 ) / np.mean( mask_val )
        
        nmse = mse / var_val
        nmse_i = mse_i / var_val
        nmse_r = mse_r / var_val
        
        print("..... Assimilation performance (validation data)")
        print(".. MSE ALL.   : %.3f / %.3f"%(mse,nmse))
        print(".. MSE ObsData: %.3f / %.3f"%(mse_r,nmse_r))
        print(".. MSE Interp : %.3f / %.3f"%(mse_i,nmse_i))
    
        trainer.test(mod, test_dataloaders=dataloaders['test'])

        # Reconstruction performance
        var_test  = np.mean( (X_test - np.mean(X_test,axis=0))**2 )
        mse = np.mean( (mod.x_rec-X_test) **2 ) 
        mse_i   = np.mean( (1.-mask_test.squeeze()) * (mod.x_rec-X_test) **2 ) / np.mean( (1.-mask_test) )
        mse_r   = np.mean( mask_test.squeeze() * (mod.x_rec-X_test) **2 ) / np.mean( mask_test )
        
        nmse = mse / var_test
        nmse_i = mse_i / var_test
        nmse_r = mse_r / var_test
        
        print("..... Assimilation performance (test data)")
        print(".. MSE ALL.   : %.3f / %.3f"%(mse,nmse))
        print(".. MSE ObsData: %.3f / %.3f"%(mse_r,nmse_r))
        print(".. MSE Interp : %.3f / %.3f"%(mse_i,nmse_i))        
        