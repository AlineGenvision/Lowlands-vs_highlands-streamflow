import torch
import scipy
import numpy as np
import torch.nn as nn
from apollo import metrics as me


### Set global model parameters
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


def load_network(len_x, len_y):

    ### Network initialisation
    net = AntecedentNET(len_x, len_y)
    net = nn.DataParallel(net)
    return net.apply(init_weights)


class RELoss(torch.nn.Module):
    def __init__(self):
        super(RELoss, self).__init__()

    def forward(self, prediction, target, psi):
        '''
        Calculate the reflective error loss as per the RELossFunc.

        Parameters:
        - prediction (Tensor): Predicted values from the model.
        - target (Tensor): Ground truth values.
        - psi (Tensor): Element-wise weights for the loss calculation.

        Returns:
        - Tensor: The computed weighted loss.
        '''
        # Compute the element-wise reflective error loss
        return torch.mean(me.RELossFunc(prediction, target, psi))

def train(net, x, y, verbose=True, loss_func_type=None, psi=None):
    ### Network training
    net = net.train()
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=0.2)
    if loss_func_type == 'Reflective':
        loss_func = RELoss()
    else:
        loss_func = torch.nn.MSELoss()
    loss_list = []
    for i in range(5000):
        y_pred = net(x.float())
        if loss_func_type is None:
            loss = loss_func(y_pred, y.float())
        else:
            loss = loss_func(y_pred, y.float(), psi)
        net.zero_grad()
        loss.backward()
        optimizer.step()
        loss_list.append(loss.data)
        if (i % 500 == 0) and verbose:
            print('epoch {}, loss {}'.format(i, loss.data))
    return net


def evaluate(net, y):

    ### Evaluate Network
    net = net.eval()
    return net(y.float()).data.cpu().numpy()
