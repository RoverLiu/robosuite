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

#--------------------------------------------
# currently it is position only estimation
#--------------------------------------------
class AbsoluteComplexFCNN:
    def __init__(self, prefix, start_pose=None):
        self.models = {}

        # get path correct
        full_path = os.path.realpath(__file__)
        path, filename = os.path.split(full_path)
        prefix = path + '/' + prefix

        # in the format of position + quaternion (x,y,z,w)
        self.current_pose = start_pose

        # load model
        self.load_model(prefix)

        self.last_input = None

        with open(prefix+'scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)

    def load_model(self, prefix):
        # define the networ
        class FC_NN(nn.Module):
            def __init__(self, input_size, output_size, hidden_size=300, dropout_rate=0.1):
                super(FC_NN, self).__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.relu1 = nn.ReLU()
                self.dropout1 = nn.Dropout(dropout_rate)
                self.fc2 = nn.Linear(hidden_size, hidden_size)
                self.relu2 = nn.ReLU()
                self.dropout2 = nn.Dropout(dropout_rate)
                self.fc2_1 = nn.Linear(hidden_size, hidden_size)
                self.relu2_1 = nn.ReLU()
                self.dropout2_1 = nn.Dropout(dropout_rate)
                self.fc2_2 = nn.Linear(hidden_size, hidden_size)
                self.relu2_2 = nn.ReLU()
                self.dropout2_2 = nn.Dropout(dropout_rate)

                
                self.fc3 = nn.Linear(hidden_size, hidden_size)
                self.relu3 = nn.ReLU()
                self.dropout3 = nn.Dropout(dropout_rate)
                self.fc4 = nn.Linear(hidden_size, output_size)
                
            def forward(self, x):
                x = self.fc1(x)
                x = self.relu1(x)
                x = self.dropout1(x)
                x = self.fc2(x)
                x = self.relu2(x)
                x = self.dropout2(x)
                x = self.fc2_1(x)
                x = self.relu2_1(x)
                x = self.dropout2_1(x)
                x = self.fc2_2(x)
                x = self.relu2_2(x)
                x = self.dropout2_2(x)
                x = self.fc3(x)
                x = self.relu3(x)
                x = self.dropout3(x)
                x = self.fc4(x)
                return x
            
        # Define model parameters
        input_size = 16 # 3+3+3
        hidden_size = 300
        # output_size = 3 # 3 digits pos
        # output_size = 4 # 7 digits quat
        output_size = 7 # 7 digits quat+pos
        dropout_rate = 0.3

        self.model = FC_NN(input_size, output_size, hidden_size=hidden_size, dropout_rate=dropout_rate)

        # Load the model
        self.model.load_state_dict(torch.load(prefix+'model.pth'))

    def get_pose(self):
        # pos_X, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w
        return self.current_pose
    
    def set_start_pose(self, start_pose):
        # pos_X, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w
        self.current_pose = start_pose
        print("current pose in the reset: {}".format(self.current_pose))


    def combined_predict(self, input_data):
        # input is contact_pos, contact_force
        # input_data = list(input_data)

        if self.current_pose == None:
            raise("Error: No initial pose given")
        
        # convert input value to required format
        data = np.concatenate([input_data, np.array(self.current_pose)])
        data = np.array([data])

        # normalize data
        data = self.scaler.transform(data)
        
        # convert to tensor
        data = torch.tensor(data, dtype=torch.float32)

        # estimate
        self.model.eval()
        with torch.no_grad():
            new_pred = self.model(data)
            new_pred = new_pred[0]

        new_pred = [float(i) for i in new_pred]
        # print('Predicted values:\n', new_pred)

        # transfer to absolute
        # position
        self.current_pose = new_pred

        return self.current_pose

# if __name__=="__main___":
# print("Path at terminal when executing this file")
# print(os.getcwd() + "\n")

# print("This file path, relative to os.getcwd()")
# print(__file__ + "\n")

# print("This file full path (following symlinks)")
# full_path = os.path.realpath(__file__)
# print(full_path + "\n")

# print("This file directory and name")
# path, filename = os.path.split(full_path)
# print(path + ' --> ' + filename + "\n")

# print("This file directory only")
# print(os.path.dirname(full_path))
# # test
# estimator = AbsoluteComplexFCNN('fc_nn/large_complex_direct/',[0.0,0.0,0.0,0.0,0.0,0.0,1.0])

# estimator.combined_predict([1,2,3,0,0,0,0,0,0])

# print(estimator.get_pose())

# estimator.combined_predict([1,2,4,0,0,0,0,0,0])

# print(estimator.get_pose())