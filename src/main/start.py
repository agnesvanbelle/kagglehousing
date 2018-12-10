
'''
 This was the provided code. I only changed it a bit. 
 '''

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import os


import pandas as pd
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


df = pd.read_json(open(os.path.join(input_dir,"train.json"), "r"))
df = df.sample(frac=1).reset_index(drop=True) # shuffle

df["num_photos"] = df["photos"].apply(len)
df["num_features"] = df["features"].apply(len)
df["num_description_words"] = df["description"].apply(lambda x: len(x.split(" ")))
df["created"] = pd.to_datetime(df["created"])
df["created_year"] = df["created"].dt.year
df["created_month"] = df["created"].dt.month
df["created_day"] = df["created"].dt.day


num_feats = ["bathrooms", "bedrooms", "latitude", "longitude", "price",
             "num_photos", "num_features", "num_description_words",
             "created_year", "created_month", "created_day"]
X = df[num_feats]
y = df["interest_level"]

print(X.head(1))


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33)

clf = RandomForestClassifier(n_estimators=1000)
clf.fit(X_train, y_train)
y_val_pred = clf.predict_proba(X_val)
print(log_loss(y_val, y_val_pred))
print((enumerate(clf.feature_importances_)))
print( sorted(enumerate(clf.feature_importances_), key = lambda x : x[1], reverse = True))
for i,f in sorted(enumerate(clf.feature_importances_), key = lambda x : x[1], reverse = True):
  print("{}\t{:2.5f}".format(X.columns[i], f))
# 
# 
# df = pd.read_json(open(os.path.join(input_dir,"test.json"), "r"))
# print(df.shape)
# df["num_photos"] = df["photos"].apply(len)
# df["num_features"] = df["features"].apply(len)
# df["num_description_words"] = df["description"].apply(lambda x: len(x.split(" ")))
# df["created"] = pd.to_datetime(df["created"])
# df["created_year"] = df["created"].dt.year
# df["created_month"] = df["created"].dt.month
# df["created_day"] = df["created"].dt.day
# X = df[num_feats]
# 
# y = clf.predict_proba(X)
# 
# 
# labels2idx = {label: i for i, label in enumerate(clf.classes_)}
# labels2idx

# sub = pd.DataFrame()
# sub["listing_id"] = df["listing_id"]
# for label in ["high", "medium", "low"]:
#     sub[label] = y[:, labels2idx[label]]
# sub.to_csv("submission_rf.csv", index=False)






