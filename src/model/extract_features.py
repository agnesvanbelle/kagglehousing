import pandas as pd
from utils.text_utils import clean_text
import pygeohash as gh

class FeatureExtractor():
  
  def __init__(self, input_file_or_dataframe, max_rows = -1, target_variable=None):
    if type(input_file_or_dataframe) == pd.core.frame.DataFrame:  # @UndefinedVariable
      self.df = input_file_or_dataframe
    else:
      self.df = pd.read_json(open(input_file_or_dataframe, "r"))
      
    if max_rows > -1:
      self.df = self.df.head(max_rows)
    
    self.target = None
    if target_variable != None:
      self.target = self.df[target_variable].copy()
      self.df.drop([target_variable], axis=1, inplace=True)

    self.extracted = False
  
  def get_target(self):
    return self.target 
  
  def get_features_all(self):
    df_features, df_featurenames = FeatureExtractor._extract_features(self.df)
    df_featurenames_sorted = sorted(df_featurenames)
    df_features = df_features.reindex(df_featurenames_sorted, axis=1)
    return df_features, df_featurenames_sorted
  
  @staticmethod
  def get_features_pred_instance(x, all_required_features):
    x_features, x_featurenames = FeatureExtractor._extract_features(x)
    
    features_only_in_x = set(x_features).difference(all_required_features)
    features_only_in_required = set(all_required_features).difference(set(x_featurenames))
    
    x_features.drop(features_only_in_x, axis = 1, inplace = True)
    x_features[list(features_only_in_required)] = pd.DataFrame([[0]*len(features_only_in_required)], index =  x_features.index)
    
    all_required_features_sorted = sorted(all_required_features)
    x_features = x_features.reindex(all_required_features_sorted, axis=1)
   
    assert(','.join(x_features.columns) == ','.join(all_required_features_sorted))
    return x_features, all_required_features_sorted
  
  def get_features(self, train_index, test_index):
    '''
    It is good habit to calculate the features separately for the train and test split.
    Otherwise, when there are features based on aggregated values (e.g. the mean of a column),
    information from the train set can slip into the test set.
    '''
    df_train, df_test = self.df.iloc[train_index], self.df.iloc[test_index]
    df_train_features, df_train_featurenames = FeatureExtractor._extract_features(df_train)
    df_test_features, df_test_featurenames = FeatureExtractor._extract_features(df_test)
    
    features_only_in_train = set(df_train_featurenames).difference(set(df_test_featurenames))
    features_only_in_test = set(df_test_featurenames).difference(set(df_train_featurenames))

    df_test_features[list(features_only_in_train)] = pd.DataFrame([[0]*len(features_only_in_train)], index = df_test_features.index)
    df_train_features[list(features_only_in_test)] = pd.DataFrame([[0]*len(features_only_in_test)], index = df_train_features.index)
    
    all_featurenames_sorted = sorted(df_train_featurenames.union(df_test_featurenames))
    
    df_train_features = df_train_features.reindex(all_featurenames_sorted, axis=1)
    df_test_features = df_test_features.reindex(all_featurenames_sorted, axis=1)
    
    assert(','.join(df_train_features.columns) == ','.join(df_test_features.columns))
    return df_train_features, df_test_features, all_featurenames_sorted
  
  @staticmethod
  def _extract_features(df):
    new_df = pd.DataFrame()    
    new_df = pd.concat([new_df, FeatureExtractor._extract_basic_integer_features(df)], axis=1)
    new_df = pd.concat([new_df, FeatureExtractor._extract_feature_features(df)], axis=1)
    new_df = pd.concat([new_df, FeatureExtractor._extract_date_features(df)], axis=1)
    new_df = pd.concat([new_df, FeatureExtractor._extract_category_features(df)], axis=1)
    new_df = pd.concat([new_df, FeatureExtractor._extract_geo_features(df)], axis=1)
    return new_df, new_df.columns
  
  @staticmethod
  def _extract_basic_integer_features(df):
    new_df = pd.DataFrame()
    new_df["num_photos"] = df["photos"].apply(len)
    new_df["num_features"] = df["features"].apply(len)
    new_df["bathrooms"] = df["bathrooms"]
    new_df["bedrooms"] = df["bedrooms"]
    new_df["num_description_words"] = df["description"].apply(lambda x: len(x.split(" ")))
    new_df["price"] = df["price"]
    return new_df
  
  @staticmethod
  def _extract_feature_features(df):
    new_df = pd.DataFrame()
    temp_features = df["features"].apply(lambda x : ['none'] if not x else [clean_text(y).replace(' ', '') for y in x])
    features_dummmified = pd.get_dummies(temp_features.apply(pd.Series).stack(), prefix="feature").sum(level=0)
    new_df = pd.concat([new_df, features_dummmified], axis=1)
    return new_df
  
  @staticmethod
  def _extract_date_features(df):
    new_df = pd.DataFrame()
    created = pd.to_datetime(df["created"])
    new_df["created_year"] = created.dt.year
    new_df["created_month"] = created.dt.month
    new_df["created_day"] = created.dt.day # day in month
    new_df["created_daynumber"] = created.apply(lambda x: x.weekday())
    new_df['created_is_weekday'] = new_df["created_daynumber"].apply(lambda x: x < 5)
    new_df['created_on_sunday'] =  new_df["created_daynumber"].apply(lambda x: x == 6)
    new_df['created_hour'] = created.apply(lambda x: x.hour)
    return new_df
  
  @staticmethod
  def _extract_category_features(df):
    return pd.concat([pd.get_dummies(df["manager_id"], prefix="man"), pd.get_dummies(df["building_id"], prefix="build")], axis=1)
  
  @staticmethod
  def _extract_geo_features(df):
    geohash_4 = df.apply(lambda x: gh.encode(x.latitude, x.longitude, precision=4), axis=1)
    geohash_5 = df.apply(lambda x: gh.encode(x.latitude, x.longitude, precision=5), axis=1)
    geohash_6 = df.apply(lambda x: gh.encode(x.latitude, x.longitude, precision=6), axis=1)
    return  pd.concat([pd.get_dummies(geohash_4, prefix = "gh4"), 
                       pd.get_dummies(geohash_5, prefix = "gh5"), 
                       pd.get_dummies(geohash_6, prefix="gh6")], axis=1)
    