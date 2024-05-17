import torch
from torch import nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.autograd import Variable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

import torch.nn.functional as F
def get_pinatubo_preds():
  np.random.seed(0)
  torch.manual_seed(0)
  device = torch.device("cpu")
  
  #Read in the dataframe from the CSV file
  df = pd.read_csv('Stdzd_Strat_Temp_AOD_Flux.csv')
  df['date'] = pd.to_datetime(df['date'])
  df

  #Extract lat, lon, and dates
  lon_vals = df['lon'].unique()
  lat_vals = df['lat'].unique()
  dates = df['date'].unique()
  
  #Create 16x48x120 tensor for Temperature data
  data_tot = torch.empty(size=(lat_vals.size,lon_vals.size,dates.size))
  data_tot_aod = torch.empty(size=(lat_vals.size,lon_vals.size,dates.size))
  data_tot_flux = torch.empty(size=(lat_vals.size,lon_vals.size,dates.size))

  for i in range(len(lat_vals)):
      for j in range(len(lon_vals)):
          temp = df[(df['lat'] == lat_vals[i]) & (df['lon'] == lon_vals[j])]
          #data_tot[i,j,:] = torch.from_numpy(np.array(temp['temperature'])).float()
          #data_tot_aod[i,j,:] = torch.from_numpy(np.array(temp['aod'])).float()
          #data_tot_flux[i,j,:] = torch.from_numpy(np.array(temp['rad_diff'])).float()
          
          data_tot[i,j,:] = torch.from_numpy(np.array(temp['temp_strat_stndzd'])).float()
          data_tot_aod[i,j,:] = torch.from_numpy(np.array(temp['aod_stndzd'])).float()
          data_tot_flux[i,j,:] = torch.from_numpy(np.array(temp['rad_diff_stndzd'])).float()

  #data_tot contains all the temperatures from all spatial points (16x48) with 120 time steps

  # This function creates sliding window and normalizes X-values (inputs to the CNN)
  def sliding_window(dates, data, seq_len, out_len): 
      x, y, start_dates = [], [], []
      
      for j in range(len(data[0,0]) - seq_len - out_len + 1): #create sliding windows
          _x = data[:,:,j:(j+seq_len)]
          _y = data[:,:, (j+seq_len):(j+seq_len+out_len)]
          x_date = dates[j]
          y_date = dates[j+seq_len]
          x.append(_x)
          y.append(_y)
          start_dates.append((x_date, y_date))
      #Normalize
      x = np.array(x)
      xmax = x.max()
      xmin = x.min()
      x_norm = x #(x - xmin) / (xmax - xmin)
      return start_dates, x_norm, np.array(y)

  #Training set A contains first 38 time steps
  #Training set B contains first 49 time steps
  train_time = 106#
  test_time = 106#
  sequence_len = 24
  output_len = 3
  x_start_dates, x_tot, y_tot =  sliding_window(dates, np.array(data_tot), sequence_len, output_len)
  x_start_dates_AOD, x_tot_AOD, y_tot_AOD = sliding_window(dates, np.array(data_tot_aod), sequence_len, output_len)
  x_start_dates_flux, x_tot_flux, y_tot_flux =  sliding_window(dates, np.array(data_tot_flux), sequence_len, output_len)

  #Grab training set A x-temperatures
  x_train_A = torch.from_numpy(x_tot[:train_time,:,:,:])

  #Grab training set A x total flux
  x_train_A_AOD = torch.from_numpy(x_tot_AOD[:train_time,:,:,:])

  #Grab training set A x total flux
  x_train_A_flux = torch.from_numpy(x_tot_flux[:train_time,:,:,:])

  #use both aod and rad flux
  x_train_A = torch.cat([x_train_A, x_train_A_AOD, x_train_A_flux], 3)

  #Grab training set A y-temperatures
  y_train_A = torch.from_numpy(y_tot[:train_time,:,:,:])
  #y_train_A = y_train_A[:,:,:,2].reshape((136,24,48,1))

  #Grab test set A x-temperatures
  x_test_A = torch.from_numpy(x_tot[test_time:,:,:,:])

  #Grab training set A x total flux
  x_test_A_AOD = torch.from_numpy(x_tot_AOD[test_time:,:,:,:])

  #Grab training set A x total flux
  x_test_A_flux = torch.from_numpy(x_tot_flux[test_time:,:,:,:])

  x_test_A = torch.cat([x_test_A, x_test_A_AOD, x_test_A_flux], 3)

  #Grab test set A y-temperatures
  y_test_A = torch.from_numpy(y_tot[test_time:,:,:,:])
  #y_test_A = y_test_A[:,:,:,2].reshape((29,24,48,1))

  #We must convert the dimensions to [1, number of time steps, 16,48] to input into the CNN
  x_train_A = torch.permute(x_train_A, (0,3,1,2))
  y_train_A = torch.permute(y_train_A, (0,3,1,2))

  x_test_A = torch.permute(x_test_A, (0,3,1,2))
  y_test_A = torch.permute(y_test_A, (0,3,1,2))

  #This function converts all sets to list of 16x48xtime tensors!!
  def list_of_tensors(tensor):
      list_tensor = []
      for i in range(len(tensor)):
          list_tensor.append(tensor[i])
      return list_tensor
    
  #For the input to the CNN we must get a list of tensors size (1,time_steps,16,48)
  x_train_A = list_of_tensors(x_train_A)
  y_train_A = list_of_tensors(y_train_A)

  x_test_A = list_of_tensors(x_test_A)
  y_test_A = list_of_tensors(y_test_A)


  class CNN_dropout(nn.Module):
      def __init__(self,dropout):
          super().__init__()
          #24 in-channels for context, 6 out-channels for horizon, kernel size 3x3
          self.dropout = nn.Dropout(dropout)
          self.fc = nn.Linear(48, 48)
          self.conv1 = nn.Conv2d(72, 24, 6, padding='same')
          self.conv2 = nn.Conv2d(24, 24, 3, padding='same')
          self.conv3 = nn.Conv2d(24, 3, 3, padding='same') #changed from 3 to 1 so it's only predicting one month ahead?

      def forward(self, x):    
          x = self.conv1(x)
          x = F.relu(x)
          x = self.conv2(x)
          x = F.relu(x)
          x = self.conv3(x)
          x = F.relu(x)

          x = self.fc(x)
          x = self.dropout(x)
          x = F.relu(x)
          
          x = self.fc(x)

          return x

  mod = CNN_dropout(0.2)
  mod.load_state_dict(torch.load("MC_dropout_thru1990.pt"))
  mod.eval()

  nmonths = (len(x_test_A))
  pred = np.zeros((1152,nmonths))
  sigma = np.zeros((1152,nmonths))
  obs = np.zeros((1152,nmonths))
  time = np.zeros((1152,nmonths))
  year = np.zeros((1152,nmonths))
  
  latlon = np.array(np.meshgrid(lon_vals, lat_vals)).reshape(2,1152).T
  lat = np.tile(latlon[:,1],nmonths)
  lon = np.tile(latlon[:,0],nmonths)
  start_yr = 1990

  n_mc = 1000

  for i in range(nmonths):
      est = np.zeros((1152,n_mc))
      for j in range(n_mc):
          temp = mod.train()(x_test_A[i].to(device))
          est[:,j] = temp[2,:,:].flatten().detach().numpy()#.cpu()
      pred[:,i] = np.mean(est,axis=1)
      sigma[:,i] = np.std(est,axis=1)
      obs[:,i] = y_test_A[i][2,:,:].flatten().detach().numpy()#.cpu()
      time[:,i] = i%12 + 1
      if i%12 == 0:
          start_yr = start_yr + 1
      year[:,i] = start_yr 
      
  out=pd.DataFrame({"Means":pred.flatten(order='F'),"Stds":sigma.flatten(order="F"),
                    "Response":obs.flatten(order="F"),"Month":time.flatten(order="F"),"Year":year.flatten(order="F"),
                  "longitude":lon,"latitude":lat})

  out['date'] = pd.to_datetime(out[['Year', 'Month']].assign(DAY=1))
  
  #Some predictions have a std of 0. We replace these with the minimum of all the predicted standard deviations in 1991, before the eruption.
  min_std = out[(out['Stds'] > 0) & (out['Year'] == 1991) & (out['Month'] < 6)]['Stds'].min()
  out.loc[out['Stds'] == 0, 'Stds'] = min_std
  dat = out.to_numpy()
  dat = dat[:,0:7]
  dat = dat.astype(float)
  return dat

dat = get_pinatubo_preds()
np.save('dat.npy', dat)