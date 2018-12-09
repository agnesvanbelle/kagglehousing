import os
import pandas as pd
import dill as pickle
from model import learn_model
from model.extract_features import FeatureExtractor
import json
import xgboost as xgb

data_dir = os.path.join(os.path.dirname(__file__), "../../data/")
input_dir = os.path.join(data_dir, "rental_listings", "input")
train_filename = os.path.join(input_dir,"train.json")
test_filename = os.path.join(input_dir,"test.json")
model_dir = os.path.join(data_dir, "models")
filename_model = os.path.abspath(os.path.join(model_dir, "model.dat"))
if not os.path.exists(model_dir):
  os.makedirs(model_dir)
  
df = pd.read_json(open(train_filename, "r"))
df = df.sample(frac=1).reset_index(drop=True) # shuffle
print(len(df))
#df = df.head(20000)

target_variable = 'interest_level'
y = df[target_variable]
X = df.drop([target_variable], axis=1)

print(X.shape)
print(y.shape)

def grid_search():
  param_combi_dict = {}
  param_combi_dict['eta'] = [0.5, 0.2]   
  param_combi_dict['max_depth'] = [ 5, 20, 50]
  param_combi_dict['gamma'] = [2, 10]
  param_combi_dict['silent'] = [1]
  param_combi_dict['colsample_bytree'] = [ 0.5, 0.8, 1]
  param_combi_dict['colsample_bylevel'] = [  0.5, 0.8, 1]
  param_combi_dict['subsample'] = [ 0.5, 0.2, 0.8]
  
  filename_grid_search_result = os.path.join(model_dir, "grid_search_result.txt")
  best_ll, best_paramcombi_string = learn_model.grid_search(X,y, [10, 100], param_combi_dict, filename = filename_grid_search_result, tytest_fraction = 0.1)
  with open(filename_grid_search_result, "w") as fo:
    fo.write('best_ll:{:2.4f}, best paramcombi:\n{:}'.format(best_ll, best_paramcombi_string))
          
def train_xvalidation():
  learn_model.train_xvalidation(X, y, n_splits=3, plot=True, num_boost_round=10)

def train_final_model():
  param = {}
  param['eta'] = 0.5
  param['max_depth'] = 20
  param['silent'] = 1
  param['colsample_by_level'] = 0.5
  param['colsample_bytree'] = 1
  param['gamma'] = 2
  param['subsample'] = 0.8
  
  model = learn_model.train(X, y, param, 10, plot=False)
  pickle.dump(model, open(filename_model, "wb"))
  print('saved model to {:s}'.format(filename_model))

def apply_final_model():
  
  model = pickle.load(open(filename_model, "rb"))
  model_feature_names = model.attr('feature_names').split('|')    
  class_to_index =  {v:int(k) for k,v in json.loads(model.attr('index_to_class')).items()}
    
  df_test = pd.read_json(open(test_filename, "r"))
  
  feature_extractor = FeatureExtractor(df_test)
  X, X_featurenames = feature_extractor.get_features_pred_instances(df_test, model_feature_names)
  y_pred = model.predict(xgb.DMatrix(X.values, feature_names = X_featurenames))
  
  filename_submission = os.path.join(data_dir, "submission_rf_agnes.csv")
  sub = pd.DataFrame()
  sub["listing_id"] = df_test["listing_id"]
  for label in ["high", "medium", "low"]:
    sub[label] = y_pred[:, class_to_index[label]]
  sub.to_csv(filename_submission, index=False)
  print ('wrote prediction subission to', filename_submission)

#grid_search()
#train_xvalidation()

#train_final_model()

apply_final_model()




