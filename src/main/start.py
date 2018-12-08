# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sys
import os
from datetime import datetime

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from utils.text_utils import clean_text

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

data_dir = "../../data/rental_listings"
input_dir = os.path.join(data_dir, "input")

print(check_output(["ls", input_dir]).decode("utf8"))

# Any results you write to the current directory are saved as output.


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.preprocessing import OneHotEncoder

df = pd.read_json(open(os.path.join(input_dir,"train.json"), "r"))
print(df.shape)
print(df.head())

print(df.columns)

import pygeohash as gh



df["num_photos"] = df["photos"].apply(len)
df["num_features"] = df["features"].apply(len)
df["num_description_words"] = df["description"].apply(lambda x: len(x.split(" ")))
#print(df["created"])

# df["weekday"] = df["created"].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S').weekday())
# print(df["weekday"]) # sunday is 6, weekday is < 5
# sys.exit(0)

df["created"] = pd.to_datetime(df["created"])
df["created_year"] = df["created"].dt.year
df["created_month"] = df["created"].dt.month
df["created_day"] = df["created"].dt.day # day in month

df = df.iloc[:1000]

print(df["price"])
print(df['building_id'])

print(len(set(df["building_id"])), len(df[df["building_id"] == '0']))
print(df['listing_id'])
print(df["created"])


print(len(df), len(set(df['manager_id'])))
print(len(df), len(set(df['listing_id'])))

print(pd.get_dummies(df[["manager_id"]]))


#feature_vectorizer = CountVectorizer(tokenizer=lambda doc: doc).fit(df["features"])
#feature_vectorizer_feature_names = feature_vectorizer.get_feature_names()
temp_df = df["features"].apply(lambda x : [clean_text(y).replace(' ', '') for y in x])
print(temp_df)
features_dummmified = pd.get_dummies(temp_df.apply(pd.Series).stack(), prefix="feature_").sum(level=0)


print(df.iloc[0])
sys.exit(0)
df['geohash']=df.apply(lambda x: gh.encode(x.latitude, x.longitude, precision=9), axis=1)

print(df['geohash'])
print(len(df), len(set(df['geohash'])))
sys.exit(0)

num_feats = ["bathrooms", "bedrooms", "latitude", "longitude", "price",
             "num_photos", "num_features", "num_description_words",
             "created_year", "created_month", "created_day"]
X = df[num_feats]
y = df["interest_level"]

print(X.shape)
print(y.shape)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33)

clf = RandomForestClassifier(n_estimators=10)
clf.fit(X_train, y_train)
y_val_pred = clf.predict_proba(X_val)
print('y_val_predicted:', y_val_pred)
print("log loss:", log_loss(y_val, y_val_pred))


df = pd.read_json(open(os.path.join(input_dir, "test.json"), "r"))
print(df.shape)
df["num_photos"] = df["photos"].apply(len)
df["num_features"] = df["features"].apply(len)
df["num_description_words"] = df["description"].apply(lambda x: len(x.split(" ")))
df["created"] = pd.to_datetime(df["created"])
df["created_year"] = df["created"].dt.year
df["created_month"] = df["created"].dt.month
df["created_day"] = df["created"].dt.day
X = df[num_feats]

y = clf.predict_proba(X)

print("y:", y)

labels2idx = {label: i for i, label in enumerate(clf.classes_)}
print(labels2idx)


sub = pd.DataFrame()
sub["listing_id"] = df["listing_id"]
for label in ["high", "medium", "low"]:
    sub[label] = y[:, labels2idx[label]]
sub.to_csv("submission_rf.csv", index=False)


