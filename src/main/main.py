import os
import pandas as pd
import dill as pickle
from model import learn_model

data_dir = os.path.join(os.path.dirname(__file__), "../../data/")
input_dir = os.path.join(data_dir, "rental_listings", "input")
train_filename = os.path.join(input_dir,"train.json")
model_dir = os.path.join(data_dir, "models")
filename_model = os.path.abspath(os.path.join(model_dir, "model.dat"))
if not os.path.exists(model_dir):
  os.makedirs(model_dir)
  
df = pd.read_json(open(train_filename, "r"))
df = df.sample(frac=1).reset_index(drop=True) # shuffle
df = df.head(10000)

target_variable = 'interest_level'
y = df[target_variable]
X = df.drop([target_variable], axis=1)

print(X.shape)
print(y.shape)

def grid_search():
  param_combi_dict = {}
  param_combi_dict['eta'] = [0.05, 0.1]   
  param_combi_dict['max_depth'] = [5, 20, 50]
  param_combi_dict['gamma'] = [10, 100]
  param_combi_dict['silent'] = [1]
  param_combi_dict['colsample_bytree'] = [0.5, 0.8]
  param_combi_dict['colsample_bylevel'] = [0.5, 0.8]
  
  learn_model.grid_search(X,y, [100,1000], param_combi_dict)

def train_xvalidation():
  learn_model.train_xvalidation(X, y, n_splits=3, plot=True, num_boost_round=10)

def train_final_model():
  param = {}
  param['num_class'] = 3
  param['eta'] = 0.1   
  param['max_depth'] = 20
  param['silent'] = 1
  param['gamma'] = 1 # for regularization
  
  model = learn_model.train(X, y, param, 100, plot=True)
  pickle.dump(model, open(filename_model, "wb"))
  print('saved model to {:s}'.format(filename_model))

grid_search()
#train_xvalidation()
#train_final_model()
# 
# model = pickle.load(open(filename_model, "rb"))
# predmat = xgb.DMatrix(np.array([feature_vector]), feature_names=feature_extractor.get_feature_names())
# return model.predict(predmat)[0]




