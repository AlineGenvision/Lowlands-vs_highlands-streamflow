'''
Artificial neural network code for reproducing the results of the 
'''

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.autograd import Variable
from apollo import streamflow as sf
from apollo import metrics as me
from apollo import mechanics as ma


### Set plotting style parameters
ma.textstyle()


### Set global model parameters
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


### Import data as dataframe and remove unclean data rows
station = 54057
filename = str(str(station) + '_lumped.csv')
rf = pd.read_csv(filename)
rf['Date'] = pd.to_datetime(rf['Date'], format='%Y-%m-%d').dt.date
rf = rf.drop(rf.index[:552])


### Identify features (with either antecedent proxies or soil moisture levels)
days = 6
features = ['Rain'] + ['Rain-' + f'{d+1}' for d in range(days)] \
            + ['Temperature'] \
            + ['Temperature-' + f'{d+1}' for d in range(days)] \
            + ['Resultant Windspeed'] \
            + ['Resultant Windspeed-' + f'{d+1}' for d in range(days)] \
            + ['Humidity'] + ['Humidity-' + f'{d+1}' for d in range(days)] \
            + ['Rain_28_Mu','Rain_90_Mu','Rain_180_Mu',
               'Temperature_28_Mu','Temperature_90_Mu','Temperature_180_Mu']
            # ['Soil Moisture ' + f'{i+1}' for i in range(4)]
targets = ['Flow']
xspace = ma.featurelocator(rf, features)
yspace = ma.featurelocator(rf, targets)


###Test/Train data split by years
yearlist = [2010+i for i in range(11)]
rftrain = rf[~pd.to_datetime(rf['Date']).dt.year.isin(yearlist)]


### Normalise features using parameters cached from the training set
norm_cache = {}
for f in features:
    rftrain[f] = ma.normalise(rftrain, f, norm_cache, write_cache=True)
    rf[f] = ma.normalise(rf, f, norm_cache, write_cache=False)


### Convert dataframe subsets to arrays and then to PyTorch variables
trnset = rftrain.to_numpy()
fullset = rf.to_numpy()
X = trnset[:,xspace].reshape(len(trnset), len(xspace)).astype(float)
Y = trnset[:,1].reshape(len(trnset), 1).astype(float)
x = torch.from_numpy(X).to(device)
y = torch.from_numpy(Y).to(device)
x, y = Variable(x), Variable(y)


### Define Neural Network structure and initialisation procedure
class AntecedentNET(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(AntecedentNET, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear_layers = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
            )
    
    def forward(self, z):
        z = self.linear_layers(z)
        return z

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif type(m) == torch.nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)


### Network initialisation
net = AntecedentNET(len(xspace), len(yspace))
net = nn.DataParallel(net)
net.apply(init_weights)


### Network training
net = net.train()
net = net.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=0.2)
loss_func = torch.nn.MSELoss()
loss_list = []
for i in range(5000):
    y_pred = net(x.float())
    loss = loss_func(y_pred, y.float())
    net.zero_grad()
    loss.backward()
    optimizer.step()
    loss_list.append(loss.data)
    if(i % 500 == 0):
        print('epoch {}, loss {}'.format(i, loss.data))


### Evaluate Network
net = net.eval()
Z = fullset[:,xspace].reshape(len(fullset), len(xspace)).astype(float)
z = torch.from_numpy(Z).to(device)
predicted = net(z.float()).data.cpu().numpy()
rf['Predicted'] = predicted
maxflow = 300
testrf = rf[pd.to_datetime(rf['Date']).dt.year.isin(yearlist)]
for df in (testrf, rf):
    sf.scatter_plot(maxflow, df, 'Predicted', 'Flow')
    print('- - - - - - - - - - - - - - -')
    print('RMSE: ' + str(me.RMSE(df['Flow'], df['Predicted'])))
    print('R\N{SUPERSCRIPT TWO}: ' + str(me.R2(df['Flow'], df['Predicted'])))
sf.year_plot(maxflow, rf, 'Predicted', 'Flow', 2007)
sf.year_plot(maxflow, rf, 'Predicted', 'Flow', 2012)


### Sensitivity Analysis using fractional increments of the maximum and minimum
### of each input variable to assess change over a baseline
sf = pd.DataFrame({'Variable':features})

x_array = []
for i in range(len(features)):
    x_array.append(np.mean(rf[features[i]]))
x_array = torch.from_numpy(np.array(x_array)).to(device).unsqueeze(0)
baseline = net(x_array.float()).data.cpu().numpy()

increment = 0.05
for k in range(int(1/increment)):
    positive_col = '+' + str(round((k+1)*increment,2))
    negative_col = '-' + str(round((k+1)*increment,2))
    sensitivities = []
    for j in range(len(features)):
        x_array = []
        for i in range(len(features)):
            mu = np.mean(rf[features[i]])
            delta = np.max(rf[features[i]])
            if i == j:
                x_array.append(mu + (k+1)*increment*(delta-mu))
            else:
                x_array.append(mu)
        x_array = torch.from_numpy(np.array(x_array)).to(device).unsqueeze(0)
        adjustment = net(x_array.float()).data.cpu().numpy()
        sensitivities.append(abs((adjustment-baseline)/baseline)[0][0])
    sf[positive_col] = sensitivities
    sensitivities = []
    for j in range(len(features)):
        x_array = []
        for i in range(len(features)):
            mu = np.mean(rf[features[i]])
            delta = np.min(rf[features[i]])
            if i == j:
                x_array.append(mu - (k+1)*increment*(mu-delta))
            else:
                x_array.append(mu)
        x_array = torch.from_numpy(np.array(x_array)).to(device).unsqueeze(0)
        adjustment = net(x_array.float()).data.cpu().numpy()
        sensitivities.append(abs((adjustment-baseline)/baseline)[0][0])
    sf[negative_col] = sensitivities