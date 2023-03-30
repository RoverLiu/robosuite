import joblib
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

class SVMModel:
    def __init__(self, prefix, start_pose=None):
        self.models = {}

        # get path correct
        full_path = os.path.realpath(__file__)
        path, filename = os.path.split(full_path)
        prefix = path + '/' + prefix

        self.models['w'] = joblib.load(prefix+'w_svm_model.joblib')
        self.models['x'] = joblib.load(prefix+'x_svm_model.joblib')
        self.models['y'] = joblib.load(prefix+'y_svm_model.joblib')
        self.models['z'] = joblib.load(prefix+'z_svm_model.joblib')
        self.models['xp'] = joblib.load(prefix+'xp_svm_model.joblib')
        self.models['yp'] = joblib.load(prefix+'yp_svm_model.joblib')
        self.models['zp'] = joblib.load(prefix+'zp_svm_model.joblib')

        # in the format of position + quaternion (x,y,z,w)
        self.current_pose = start_pose

        # print("current pose in the beginning: {}".format(self.current_pose))


        self.last_input = None

        with open(prefix+'scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)

    def single_predict(self, model_name, input_data):
        if model_name not in self.models:
            raise ValueError(f"Invalid model name: {model_name}")
        model = self.models[model_name]
        return model.predict(input_data)
    
    def get_pose(self):
        return self.current_pose
    
    def set_start_pose(self, start_pose):
        self.current_pose = start_pose
        print("current pose in the reset: {}".format(self.current_pose))


    def combined_predict(self, input_data):
        input_data = list(input_data)

        if self.current_pose == None:
            raise("Error: No initial pose given")
        
        # already calcullated
        if self.last_input == input_data:
            self.last_input = input_data
            return self.current_pose
        else:
            self.last_input = input_data

        # convert input value to required format
        data = input_data[3:]

        data.append(input_data[0]-self.current_pose[0])
        data.append(input_data[1]-self.current_pose[1])
        data.append(input_data[2]-self.current_pose[2])

        data = np.array([data])

        # normalize data
        data = self.scaler.transform(data)
        
        # predict all values 
        relatives = {}

        for model_name in self.models.keys():
            relatives[model_name] = self.single_predict(model_name, data)

        # transfer to absolute
        absolute = self.current_pose
        # position
        absolute[0] += relatives['xp']
        absolute[1] += relatives['yp']
        absolute[2] += relatives['zp']
        # orientation
        # print("current pose: {}".format(self.current_pose))
        # print(absolute)
        [w,x,y,z] = self.relative_quat(absolute[3:], [ relatives['x'], relatives['y'], relatives['z'], relatives['w']])
        absolute[3] = x
        absolute[4] = y
        absolute[5] = z
        absolute[6] = w

        # convert to the format needed
        formated = [i[0] for i in absolute]

        # update
        self.current_pose = formated
        # print("current pose changed to: {}".format(self.current_pose))


        return formated

    def relative_quat(self, current, relative):
        '''quaternion in xyzw format'''
        # rotation (inverse)
        q_0_x = -relative[0]
        q_0_y = -relative[1]
        q_0_z = -relative[2]
        q_0_w = relative[3]

        # current
        q_1_x = current[0]
        q_1_y = current[1]
        q_1_z = current[2]
        q_1_w = current[3]

        # relative
        w = q_0_w*q_1_w - (q_0_x*q_1_x + q_0_y*q_1_y + q_0_z*q_1_z)
        x = q_0_w*q_1_x + q_1_w*q_0_x + q_0_y*q_1_z - q_0_z*q_1_y
        y = q_0_w*q_1_y + q_1_w*q_0_y + q_0_z*q_1_x - q_0_x*q_1_z
        z = q_0_w*q_1_z + q_1_w*q_0_z + q_0_x*q_1_y - q_0_y*q_1_x
        
        # normalize
        norm = np.linalg.norm([w,x,y,z])

        return [w,x,y,z]/norm

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
# test
estimator = SVMModel('svm/small/',[0.0,0.0,0.0,0.0,0.0,0.0,1.0])

estimator.combined_predict([1,2,3,0,0,0,0,0,0])

print(estimator.get_pose())

estimator.combined_predict([1,2,3,0,0,0,0,0,0])

print(estimator.get_pose())