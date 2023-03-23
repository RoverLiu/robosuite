import joblib

class SVMModel:
    def __init__(self, prefix, start_pose):
        self.models = {}
        self.models['w'] = joblib.load(prefix+'w_svm_model.joblib')
        self.models['x'] = joblib.load(prefix+'x_svm_model.joblib')
        self.models['y'] = joblib.load(prefix+'y_svm_model.joblib')
        self.models['z'] = joblib.load(prefix+'z_svm_model.joblib')
        self.models['xp'] = joblib.load(prefix+'xp_svm_model.joblib')
        self.models['yp'] = joblib.load(prefix+'yp_svm_model.joblib')
        self.models['zp'] = joblib.load(prefix+'zp_svm_model.joblib')

        self.current_pose = start_pose

    def single_predict(self, model_name, input_data):
        if model_name not in self.models:
            raise ValueError(f"Invalid model name: {model_name}")
        model = self.models[model_name]
        return model.predict(input_data)
    
    def combined_predict(self, input_data):
        # predict all values 
        relatives = {}

        for model_name in self.models.keys():
            relatives[model_name] = self.single_predict(model_name, input_data)

        # transfer to absolute
        absolute = self.current_pose
        # position
        absolute[0] += relatives['xp']
        absolute[1] += relatives['yp']
        absolute[2] += relatives['zp']
        # orientation
        [w,x,y,z] = self.relative_quat(absolute[3:], [relatives['w'], relatives['x'], relatives['y'], relatives['z']])
        absolute[3] = x
        absolute[4] = y
        absolute[5] = z
        absolute[6] = w
        
        # update
        self.current_pose = absolute

        return absolute

    def relative_quat(current, relative):
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
        # print([w,x,y,z])
        return [w,x,y,z]


