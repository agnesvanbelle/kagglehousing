from sklearn.feature_extraction import DictVectorizer
import pandas as pd
import datetime as dt
from collections import OrderedDict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import itertools
import numpy as np
import sys
from utils.text_utils import clean_text

from sklearn.preprocessing import OneHotEncoder


class FeatureExtractor():
  
  def __init__(self, input_file, max_rows = -1, target_variable=None):
    df = pd.read_json(open(input_file, "r"))
    self.df = df
    if max_rows > -1:
      self.df = self.df.iloc[:max_rows]
    self.target = None
    if target_variable != None:
      self.target = df[target_variable].copy()
      df.drop([target_variable], axis=1, inplace=True)
      
    self.new_df = pd.DataFrame()
  
  def get_target(self):
    return self.target 
  
  def get_features(self):
    self._extract_features()
    
    #self.new_df =  self.new_df.replace(float('nan'), -1)
    #mask = self.new_df.apply(lambda x : np.any(pd.isnull(x)) != True, axis=1) 
    #self.new_df = self.new_df[mask]
    
    return self.new_df
  
  def _extract_features(self):
    self.extract_basic_integer_features()
    self.extract_feature_features()
    
  def extract_basic_integer_features(self):

    self.new_df["num_photos"] = self.df["photos"].apply(len)
    self.new_df["num_features"] = self.df["features"].apply(len)
    self.new_df["num_description_words"] = self.df["description"].apply(lambda x: len(x.split(" ")))
    #print(df["created"])
    
    # df["weekday"] = df["created"].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S').weekday())
    # print(df["weekday"]) # sunday is 6, weekday is < 5
    # sys.exit(0)
    
    created = pd.to_datetime(self.df["created"])
    self.new_df["created_year"] = created.dt.year
    self.new_df["created_month"] = created.dt.month
    self.new_df["created_day"] = created.dt.day # day in month
  
  def extract_feature_features(self):

    temp_features = self.df["features"].apply(lambda x : ['none'] if not x else [clean_text(y).replace(' ', '') for y in x])
    features_dummmified = pd.get_dummies(temp_features.apply(pd.Series).stack(), prefix="feature_").sum(level=0)
    self.new_df = pd.concat([self.new_df, features_dummmified], axis=1)
    