import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import os
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

#Dropout neural network model we use for the base forecaster.
class MLP(nn.Module):
  def __init__(self, d):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(d, 64),
      nn.ReLU(),
      nn.Linear(64, 32),
      nn.ReLU(),
      nn.Linear(32, 1),
    )
    self.dropout = nn.Dropout(0.25)
  def forward(self, x):
    x = self.dropout(x)
    x = self.layers(x)
    return x

def train(model, num_epochs, optimizer, loss_function, data_loader):
  losses = torch.zeros(num_epochs)
  for epoch in range(0, num_epochs): 
    current_loss = 0.0
    for i, (inputs, targets) in enumerate(data_loader):
      inputs, targets = inputs.float(), targets.float()
      targets = targets.reshape((targets.shape[0], 1))
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = loss_function(outputs, targets)
      loss.backward()
      optimizer.step()
      current_loss += loss.item()
    losses[epoch] = current_loss/len(data_loader.dataset)
    if epoch % 5 == 0:
      print(str(epoch) + ": " + str(losses[epoch].item()))
  print('Training process has finished.');
  plt.plot(losses)
  plt.show()

def get_loader(X,y,batch_size):
  X,y = torch.tensor(X), torch.tensor(y)
  if torch.cuda.is_available():
    X,y = X.cuda(), y.cuda()
  return DataLoader(TensorDataset(X,y),batch_size=batch_size,shuffle=False)

#Trains the model on the data trainX, trainy.
def get_model(trainX, trainy,  
              num_epochs = 50, lr = 1e-4, batch_size = 50):
  train_loader = get_loader(trainX,trainy,batch_size)
  d = len(train_loader.dataset[0][0])
  mlp = MLP(d)
  if torch.cuda.is_available():
    mlp = mlp.cuda()
  loss_function = nn.L1Loss()
  optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)
  train(mlp, num_epochs, optimizer, loss_function, train_loader)
  return mlp

#outputs mean and std of dropout predictions from model
def test(model, X, y, dropout_iters = 100, test_batch_size = 5000):
  data_loader = get_loader(X,y,test_batch_size)
  model.train()
  preds_mat = torch.zeros(len(data_loader.dataset), dropout_iters)
  
  for i in range(dropout_iters):
    preds = []
    with torch.no_grad():
      for inputs, targets in data_loader:
        inputs, targets = inputs.float(), targets.float()
        targets = targets.reshape((targets.shape[0], 1))
        outputs = model(inputs)
        preds.extend(outputs.detach().cpu().reshape(-1))
    preds = torch.tensor(preds)
    preds_mat[:,i] = preds
  means = torch.mean(preds_mat, dim = 1)
  stds = torch.std(preds_mat, dim = 1)
  return np.array(means), np.array(stds)

#Trains model on california housing dataset where data is split according to a random seed,
#obtains dropout predictions on test data.
def get_california_preds(seed):
  torch.manual_seed(seed)
  np.random.seed(seed)
  
  housing = fetch_california_housing()
  X = housing['data']
  y = housing['target']
  trainX, testX, trainy, testy = train_test_split(X,y,test_size=0.15)
  trainX, validX, trainy, validy = train_test_split(trainX, trainy, test_size = 0.15)

  batch_size = 64
  num_epochs = 50
  lr = 1e-4
  test_batch_size = 5000
  dropout_iters = 100
  
  mlp = get_model(trainX, trainy, num_epochs, lr, batch_size)

  test_means, test_stds = test(mlp, testX, testy, dropout_iters, test_batch_size)
  valid_means, valid_stds = test(mlp, validX, validy, dropout_iters, test_batch_size)
  
  valid_data = np.hstack([valid_means.reshape(-1,1), valid_stds.reshape(-1,1), validy.reshape(-1,1)])
  test_data = np.hstack([test_means.reshape(-1,1), test_stds.reshape(-1,1), testy.reshape(-1,1)])
  
  return valid_data, test_data

if not(os.path.exists('splits')):
  os.mkdir('splits')

seeds = np.arange(10)
for seed in seeds:
  valid_data, test_data = get_california_preds(seed)
  np.savez('splits/split_' + str(seed) + '.npz', valid_data = valid_data, test_data = test_data)