import joblib
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import copy
# from sklearn import svm

from torch.utils.data import TensorDataset

# temporal covnet
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import math 
from torch.autograd import Variable


# %% model definition
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    


class TCNN_Simple(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout, FCNN_hidden_size):
        super(TCNN_Simple, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.fc1 = nn.Linear(num_channels[-1], FCNN_hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(FCNN_hidden_size, FCNN_hidden_size)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)        
        self.fc3 = nn.Linear(FCNN_hidden_size, FCNN_hidden_size)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)
        self.fc4 = nn.Linear(FCNN_hidden_size, output_size)

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        out = self.tcn(inputs)  # input should have dimension (N, C, L)
        
        x = self.fc1(out[:,:,-1])
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        x = self.fc4(x)
        return x


# %%
#--------------------------------------------
# currently it is position only estimation
#--------------------------------------------
class TCN_manual:
    def __init__(self, prefix, start_pose=None, sequence = 50, input_size = 16, y_coefficient = [2, 1, 1, 5, 7, 7, 7]):
        self.models = {}

        # get path correct
        full_path = os.path.realpath(__file__)
        path, filename = os.path.split(full_path)
        prefix = path + '/' + prefix

        # define the size of sequence 
        self.sequence = sequence
        self.input_size = input_size

        # in the format of position + quaternion (x,y,z,w)
        self.current_pose = start_pose

        # load model
        self.load_model(prefix)

        self.last_input = None

        self.y_coefficient = y_coefficient

        # load pre-process dat
        with open(prefix+'scaler_x.pkl', 'rb') as f:
            self.scaler_x = pickle.load(f)

    def load_model(self, prefix):
            
        # Define model parameters
        # Given groups=1, weight of size [25, 16, 4], expected input[64, 10, 16] to have 16 channels, but got 10 channels instead
        # Define model parameters
        hidden_size = 100
        output_size = 7 # 7 digits quat+pos
        levels =  int(math.log2(self.sequence)+1)# consider this size of sequence
        channel_size = [hidden_size] * levels    
        kernel_size = 6
        dropout = 0.25
        FCNN_hidden_size = 300

        self.model = TCNN_Simple(self.input_size, output_size, channel_size, kernel_size, dropout, FCNN_hidden_size)

        # Load the model
        self.model.load_state_dict(torch.load(prefix+'model.pth'))

    def get_pose(self):
        return self.current_pose
    
    def set_start_pose(self, start_pose):
        self.current_pose = start_pose
        print("current pose in the reset: {}".format(self.current_pose))


    def combined_predict(self, input_data):

        if type(self.current_pose) == type(None):
            raise("Error: No initial pose given")
        
        if type(self.last_input) == type(None):
            input_data = list(input_data)
            input_data = input_data + self.current_pose
            input_data = self.scaler_x.transform(np.array([input_data]))
            self.last_input = [input_data[0] for i in range(self.sequence)]
        else:
            input_data = list(input_data)
            input_data = input_data + self.current_pose
            input_data = self.scaler_x.transform(np.array([input_data]))
            print("size of last input {}, size of new input: {}".format(self.last_input[1:,:].shape, np.array(input_data).shape))
            self.last_input = np.concatenate([self.last_input[1:,:], np.array(input_data)])

        self.last_input = np.array(self.last_input)
        print("last input shape: {}".format(self.last_input.shape))

        # print(self.last_input)

        # convert input value to required format
        data = self.last_input
        print("data size: {}".format(data.shape))

        # convert to tensor
        data = torch.tensor(np.array([data]), dtype=torch.float32)
        print("data size (tensor): {}, format: {}".format(data.shape, type(data)))

        data = data.view(-1, self.input_size, self.sequence)

        print("shape of data: {}".format(data.shape))

        data = Variable(data)

        # estimate
        self.model.eval()
        with torch.no_grad():
            new_pred = self.model(data)
            new_pred = new_pred[0]

        new_pred = [float(i) for i in new_pred]
        new_pred = np.divide(new_pred, self.y_coefficient)
        print('Predicted values ori:\n', new_pred)        

        self.current_pose = list(new_pred)

        # assume the quat/orientation remains the same

        return self.current_pose


# # test
# estimator = TCN_manual('tcnn/manual_y/',[0.0,0.0,0.0,0.0,0.0,0.0,1.0])

# estimator.combined_predict([1,2,3,0,0,0,0,0,0])

# print(estimator.get_pose())

# estimator.combined_predict([1,2,4,0,0,0,0,0,0])

# print(estimator.get_pose())