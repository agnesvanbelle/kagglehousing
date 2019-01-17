import xgboost as xgb
#from multiprocessing import cpu_count
from sklearn.metrics import log_loss, confusion_matrix, cohen_kappa_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from model.extract_features import FeatureExtractor
import sys
import math
import itertools
import json
import seaborn as sn

default_param = {}
default_param['objective'] = 'multi:softprob'
default_param['num_class'] = 3
default_param['eval_metric'] = ['merror', 'mlogloss']
default_param['eta'] = 0.1   
default_param['max_depth'] = 20
default_param['silent'] = 1
default_param['gamma'] = 1 # for regularization
default_param['subsample'] = 0.8
default_param['colsample_bytree'] = 0.5 # subsample ratio of features when constructing each tree.
default_param['colsample_bylevel'] = 0.5 # subsample ratio of features for each split
default_param['min_child_weight'] = 2
#default_param['nthread'] = (cpu_count()-1)*2 if cpu_count() != None else 4


def _get_random_prediction(y_train, y_test, labels_sorted):
  sample_prob = {k:v/float(len(y_train)) for k,v in dict(y_train.value_counts()).items()}
  random_predicted_labels = np.random.choice(labels_sorted, size=len(y_test), replace=True, p = [sample_prob[c] for c in labels_sorted])
  df_pred = pd.DataFrame(np.zeros((len(y_test), 3)), columns = labels_sorted)
  for i, label in enumerate(random_predicted_labels):
    df_pred.ix[i,label] = 1
  return df_pred

def grid_search(X, y, num_boost_rounds_list, param_list_dict, n_splits=3, plot=False, verbose_eval = False, filename = None,
                test_fraction = 0.1):
  '''
  Do a grid search given a list containing the num_boost_round parameters (num_boost_rounds) that should be tested 
  and a dictionary (param_list_dict) that has a key the model parameter (see default_param above) 
  and as value a list of values that should be used for that parameter. All num_boost_round & parameter combinations
  are checked using cross-validation, using he function train_xvalidation below. Returns the best log loss
  value and a string representation of the best model parameters found. If a filename is given it
  writes the latter two values each time they are updated to that filename.
  '''
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
    _, avg_ll_model, _, _, _, _, _ = train_xvalidation(X, y, param = this_params, num_boost_round= param_combi[0], n_splits = n_splits, 
                                                 plot = plot, verbose_eval = verbose_eval, test_fraction = test_fraction)
    if avg_ll_model < best_ll:
      best_ll = avg_ll_model
      best_paramcombi_index = i
      if filename != None:
        with open(filename, "w") as fo:
          fo.write('best_ll:{:2.4f}, best paramcombi:\n{:}'.format(best_ll, get_paramcombi_string(i)))
  
  print('best ll: {:2.2f}, best paramcombi:\n{:s}'.format(best_ll, get_paramcombi_string(best_paramcombi_index)))
  return best_ll, get_paramcombi_string(best_paramcombi_index)

def train(X, y, param, num_boost_round, plot = True, verbose_eval = True):
  '''
  Train a single model. Returns the model (an Xgboost Booster)
  '''
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
  
def train_xvalidation(X, y, param = default_param, num_boost_round = 100, 
                            n_splits=3, plot=True, verbose_eval = True, test_fraction = 0.1, make_stats = False):    
  '''
  Do several cross-validation rounds, using shuffle-splits. Returns the average log loss of the
  baseline (calculated using random predictions per split), average log loss of the model, and
  average relative log loss reduction of the model over the baseline. The averages are over the splits. 
  '''
  relative_reductions = []
  ll_baselines = []
  ll_models = []
  y_true_all = np.array([], dtype='str')
  y_pred_all = np.array([], dtype='str')
  
  for k in default_param:
    if not k in param.keys():
      param[k] = default_param[k]
  
  skf = StratifiedShuffleSplit(n_splits=n_splits,  train_size = 1 - test_fraction, test_size = test_fraction)
  skf.get_n_splits(X, y)

  for train_index, test_index in skf.split(X, y):
    X_train, X_test, featurenames = FeatureExtractor(X).get_features(train_index, test_index)
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    y_true_all = np.concatenate((y_true_all, y_test))
    
    labels_sorted = list(sorted(set(y)))

    random_prediction = _get_random_prediction(y_train, y_test, labels_sorted)
    ll_baseline =  log_loss(y_test, random_prediction.values, labels = labels_sorted)
    ll_baselines.append(ll_baseline)
    
    class_to_index = {k:v for k,v in zip(labels_sorted, range(len(labels_sorted)))} 
    index_to_class = {v:k for k,v in zip(labels_sorted, range(len(labels_sorted)))} 
    
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
    
    y_pred_labeled = np.array([index_to_class[i] for i in np.argmax(y_pred, 1)])
    y_pred_all = np.concatenate((y_pred_all, y_pred_labeled))
    
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
  
  if make_stats:
    _make_stats(ll_baseline, ll_model, relative_reduction, y_true_all, y_pred_all, index_to_class, labels_sorted, plot = True)
  return avg_ll_baseline, avg_ll_model, avg_relative_reduction, y_true_all, y_pred_all, index_to_class, labels_sorted

def _make_stats(ll_baseline, ll_model, relative_reduction, y_true, y_pred, index_to_class, labels_sorted, plot = True):
  
  
  print('y_true:',y_true[:5,])
  print('y_pred:',y_pred[:5,])
  
  print('len y_true:',len(y_true))
  
  print('\tll baseline: %2.2f, ll model: %2.2f, relative reduction: %2.2f perc' % (ll_baseline, ll_model, relative_reduction))
  
  print('labels_sorted:', labels_sorted)
  cm = confusion_matrix(y_true, y_pred, labels = labels_sorted)
  cm_rel = np.nan_to_num(100 * (cm.T / cm.sum(axis=1)).T)
  
  print('cm:\n', cm)
  print('cm_rel:\n', cm_rel)
  
  ck = cohen_kappa_score(y_true, y_pred, labels = labels_sorted)
  print("cohens kappa:",ck)
  
  if plot:
    _make_cf_matrix_plot(cm, labels_sorted)
    plt.gcf().clear()
    plt.close()
    _make_cf_matrix_plot(cm_rel, labels_sorted)

def _make_cf_matrix_plot(cm, labels):
  df_cm = pd.DataFrame(cm, columns = labels, index = labels)#, index = [i for i in "ABCDEFGHIJK"],columns = [i for i in "ABCDEFGHIJK"])
  plt.figure(figsize = (10,7))
  sn.set(font_scale=1.4)#for label size
  sn.heatmap(df_cm, annot=True, cmap = sn.cm.rocket_r)
  
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.tight_layout()
  plt.show()
