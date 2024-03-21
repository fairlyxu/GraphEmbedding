
import pickle
import traceback

class FeaturesProcess():
    def __init__(self, features_label_file=""):
        self.features_label_file = features_label_file
        self.features_label = {}
        try:
            with open(features_label_file, 'rb') as f:
                self.features_label = pickle.load(f)
        except:
            traceback.print_exc()

    def save_feature_file(self):
        # save
        with open(self.features_label_file, 'wb') as f:
            pickle.dump(self.features_label, f)

    def fit_transform(self,f_key,f_value):
        # init config file
        if f_key not in self.features_label:
            tmp = {}
            tmp["e_lable"] = {}
            tmp["r_lable"] = {}
            tmp["max"] = -1
            self.features_label[f_key] = tmp
        if f_value not in self.features_label[f_key]["e_lable"]:
            index = self.features_label[f_key]["max"] + 1
            self.features_label[f_key]["max"] = index
            self.features_label[f_key]["e_lable"][f_value] = index
            self.features_label[f_key]["r_lable"][index] = f_value
        return self.features_label[f_key]["e_lable"].get(f_value)

    def transform(self,f_key,f_value):
        return self.features_label[f_key]["e_lable"].get(f_value,0)

    def inverse_transform(self,f_key,f_value):
        return self.features_label[f_key]["r_lable"].get(f_value)


