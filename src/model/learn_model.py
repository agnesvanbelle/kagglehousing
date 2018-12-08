import xgboost as xgb
from multiprocessing import cpu_count
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from model.extract_features import FeatureExtractor
import sys
import math
import itertools
import json

default_param = {}
default_param['objective'] = 'multi:softprob'
default_param['num_class'] = 3
default_param['eval_metric'] = ['merror', 'mlogloss']
default_param['eta'] = 0.1   
default_param['max_depth'] = 20
default_param['silent'] = 1
default_param['gamma'] = 1 # for regularization
default_param['colsample_bytree'] = 0.5 # subsample ratio of features when constructing each tree.
default_param['colsample_bylevel'] = 0.5 # subsample ratio of features for each split
default_param['min_child_weight'] = 2
default_param['nthread'] = cpu_count() if cpu_count() != None else 4


def _get_random_prediction(y_train, y_test, labels_sorted):
  sample_prob = {k:v/float(len(y_train)) for k,v in dict(y_train.value_counts()).items()}
  random_predicted_labels = np.random.choice(labels_sorted, size=len(y_test), replace=True, p = [sample_prob[c] for c in labels_sorted])
  df_pred = pd.DataFrame(np.zeros((len(y_test), 3)), columns = labels_sorted)
  for i, label in enumerate(random_predicted_labels):
    df_pred.ix[i,label] = 1
  return df_pred

def grid_search(X, y, num_boost_rounds_list, param_list_dict, n_splits=3, plot=False, verbose_eval = False):
  keys_params = sorted(list(param_list_dict.keys()))
  keys_to_index = {k:v for k,v in zip(keys_params, range(len(keys_params)))}
  combis = list(itertools.product(num_boost_rounds_list, *[param_list_dict[k] for k in keys_params]))
  
  best_ll = sys.maxsize
  best_paramcombi_index = -1
  
  def get_paramcombi_string(index):
    s = '\tnum_boost_rounds: {:d}\n'.format(combis[index][0])
    for k in keys_params:
      s += '\t{:s}: {:}\n'.format(k, combis[index][keys_to_index[k]+1])
    return s.strip()
    
  for i, param_combi in enumerate(combis):
    print('doingparam combi {:d} of {:d}:\n{:s}'.format(i, len(combis), get_paramcombi_string(i)))
    
    this_params = {k : param_combi[keys_to_index[k] + 1] for k in keys_params}
    for k in default_param:
      if not k in keys_params:
        this_params[k] = default_param[k]
    _, avg_ll_model, _ = train_xvalidation(X, y, param = this_params, num_boost_round= param_combi[0], n_splits = n_splits, 
                                                 plot = plot, verbose_eval = verbose_eval)
    if avg_ll_model < best_ll:
      best_ll = avg_ll_model
      best_paramcombi_index = i
  
  print('best ll: {:2.2f}, best paramcombi:\n{:s}'.format(best_ll, get_paramcombi_string(best_paramcombi_index)))
  return best_ll, get_paramcombi_string(best_paramcombi_index)
  
def train(X, y, param, num_boost_round, plot = True, verbose_eval = True):
  
  X_train, featurenames = FeatureExtractor(X).get_features_all()
  labels_sorted = list(sorted(set(y)))
  class_to_index = {k:v for k,v in zip(labels_sorted, range(len(labels_sorted)))} 
  index_to_class = {v:k for k,v in zip(labels_sorted, range(len(labels_sorted)))} 
  y_train_numeric = [class_to_index[k] for k in y]
  
  for k in default_param:
    if not k in param.keys():
      param[k] = default_param[k]
        
  xgdmat = xgb.DMatrix(X_train.values, label = y_train_numeric, feature_names = featurenames)
  final_gb = xgb.train(param, xgdmat, evals = [(xgdmat, "trainset")], verbose_eval = verbose_eval, num_boost_round = num_boost_round)
    
  if plot:
    _fig, ax = plt.subplots(1,1,figsize=(20,30))
    xgb.plot_importance(final_gb, color='red',  ax=ax, max_num_features=25)
    plt.show()
  print('final_gb.feature_names:', final_gb.feature_names)
  if hasattr(final_gb, 'feature_names'): 
    final_gb.set_attr(feature_names = '|'.join(final_gb.feature_names))
  final_gb.set_attr(index_to_class = json.dumps(index_to_class))
  return final_gb
  
def train_xvalidation(X, y, param=default_param, num_boost_round=100, 
                            n_splits=3, plot=True, verbose_eval = True):    

  relative_reductions = []
  ll_baselines = []
  ll_models = []
  
  skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
  skf.get_n_splits(X, y)

  for train_index, test_index in skf.split(X, y):
    X_train, X_test, featurenames = FeatureExtractor(X).get_features(train_index, test_index)
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    labels_sorted = list(sorted(set(y)))

    random_prediction = _get_random_prediction(y_train, y_test, labels_sorted)
    ll_baseline =  log_loss(y_test, random_prediction.values, labels = labels_sorted)
    ll_baselines.append(ll_baseline)
    
    class_to_index = {k:v for k,v in zip(labels_sorted, range(len(labels_sorted)))} 
    y_train_numeric = [class_to_index[k] for k in y_train]
    y_test_numeric = [class_to_index[k] for k in y_test]
    
    xgdmat = xgb.DMatrix(X_train.values, label = y_train_numeric, feature_names = featurenames)
    valmat = xgb.DMatrix(X_test.values, label = y_test_numeric,  feature_names = featurenames)
    final_gb = xgb.train(param, xgdmat, evals = [(xgdmat, "trainset"), (valmat, "testset")], verbose_eval = verbose_eval, 
                         num_boost_round = num_boost_round)
    
    if plot:
      _fig, ax = plt.subplots(1,1,figsize=(20,30))
      xgb.plot_importance(final_gb, color='red',  ax=ax, max_num_features=25)
      plt.show()

    testmat = xgb.DMatrix(X_test.values, label = y_test_numeric, feature_names = featurenames)
    y_pred = final_gb.predict(testmat)
    
    ll_model = log_loss(y_test, y_pred, labels = labels_sorted)
    ll_models.append(ll_model)
    relative_reduction = (ll_baseline - ll_model) / (ll_baseline / 100.0)
    if ll_baseline <=  math.pow(10, -5):
      relative_reduction = 0
    relative_reductions.append(relative_reduction)
    
    print('\tll baseline: %2.2f, ll model: %2.2f, relative reduction: %2.2f perc' % (ll_baseline, ll_model, relative_reduction))
      
  avg_ll_baseline, avg_ll_model, avg_relative_reduction = np.mean(ll_baselines), np.mean(ll_models), np.mean(relative_reductions)
  print('done {:d} splits. avg. ll baseline: {:2.2f}, avg. ll model: {:2.2f}, avg. relative reduction: {:2.2f}'
        .format(n_splits, avg_ll_baseline, avg_ll_model, avg_relative_reduction))
  
  return avg_ll_baseline, avg_ll_model, avg_relative_reduction

