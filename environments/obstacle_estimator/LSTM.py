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
class LSTM_Simple(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, bidirectional=False):
        super(LSTM_Simple, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # print(x.shape)
        out, (h_n, c_n) = self.lstm(x)
        # print(out.shape)
        # print(h_n.shape)
        # print(c_n.shape)
        out = self.fc(out[:,-1,:])
        return out
    


# %%
#--------------------------------------------
# currently it is position only estimation
#--------------------------------------------
class LSTM_more_hidden:
    def __init__(self, prefix, start_pose=None, sequence = 10, input_size = 16):
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

    

        # load pre-process dat
        with open(prefix+'scaler.pkl', 'rb') as f:
            self.scaler_x = pickle.load(f)

        # with open(prefix+'scaler_y.pkl', 'rb') as f:
        #     self.scaler_y = pickle.load(f)
    
    def load_model(self, prefix):
            
        # Define model parameters
        # Given groups=1, weight of size [25, 16, 4], expected input[64, 10, 16] to have 16 channels, but got 10 channels instead
        # Define model parameters
        # input_size = 16 # 3+3+3
        hidden_size = 100
        # output_size = 3 # 3 digits pos
        # output_size = 4 # 7 digits quat
        output_size = 7 # 7 digits quat+pos
        num_layers = self.sequence # consider this size of sequence


        self.model = LSTM_Simple(self.input_size, output_size, hidden_size, num_layers)

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

        # data = data.view(-1, self.input_size, self.sequence)

        print("shape of data: {}".format(data.shape))


        # estimate
        self.model.eval()
        with torch.no_grad():
            new_pred = self.model(data)
            new_pred = new_pred[0]

        new_pred = [float(i) for i in new_pred]
        print('Predicted values ori:\n', new_pred)

        # transfer to original
        # new_pred = self.scaler_y.inverse_transform([new_pred])
        # print('Predicted values transformed:\n', new_pred)
        

        self.current_pose = new_pred

        # assume the quat/orientation remains the same

        return self.current_pose


# # test
# estimator = LSTM_more_hidden('lstm/direct_simple_more_hidden/',[0.0,0.0,0.0,0.0,0.0,0.0,1.0])

# estimator.combined_predict([1,2,3,0,0,0,0,0,0])

# print(estimator.get_pose())

# estimator.combined_predict([1,2,4,0,0,0,0,0,0])

# print(estimator.get_pose())