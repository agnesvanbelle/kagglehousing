from flask import Flask, jsonify
from flask import request
from cgi import escape
import os
import dill as pickle
from main.main import filename_model
from model.extract_features import FeatureExtractor
import pandas as pd
import xgboost as xgb
import json
from service.interest_prediction import InterestPrediction

model = None
model_feature_names = None
index_to_class = None

app = Flask('Apartment interest predictor')

# HTTP Errors handlers
@app.errorhandler(404)
def url_error(e):
  return """
  Wrong URL!
  <pre>{}</pre>""".format(e), 404

@app.errorhandler(500)
def server_error(e):
  return """
  An internal error occurred: <pre>{}</pre>
  See logs for full stacktrace.
  """.format(e), 500

def load_model():
  global model
  global index_to_class
  global model_feature_names
  if os.path.exists(filename_model):
    model = pickle.load(open(filename_model, "rb"))
    model_feature_names = model.attr('feature_names').split('|')    
    index_to_class =  {int(k):v for k,v in json.loads(model.attr('index_to_class')).items()}
  else:
    raise FileNotFoundError("Model not found: {:s}".format(filename_model))
    

@app.route('/interest_prediction', methods=['GET'])
def get_prediction():
  '''
  bathrooms: number of bathrooms 
  bedrooms: number of bathrooms 
  building_id 
  created 
  description 
  display_address 
  features: a list of features about this apartment 
  latitude 
  listing_id 
  longitude 
  manager_id 
  photos: a list of photo links. You are welcome to download the pictures yourselves from renthop's site, but they are the same as imgs.zip. 
  price: in USD 
  street_address 
  '''
  d = {}
  
  d['bathrooms'] = request.args.get('bathrooms', type = int)
  d['bedrooms'] = request.args.get('bedrooms', type = int)
  d['building_id'] = request.args.get('building_id', type = str, default = '0')
  d['description'] = request.args.get('description', type = str, default= '')
  #display_address = request.args.get('display_address', type = str)
  d['features'] = request.args.getlist('feature', type = str)
  d['latitude'] = request.args.get('latitude', type = float, default = 0)
  d['longitude'] = request.args.get('longitude', type = float, default = 0)
  #listing_id = request.args.get('isting_id', type = int)
  d['manager_id'] = request.args.get('manager_id', type = str, default = '')
  d['photos'] = request.args.getlist('photo', type = str)
  d['price'] = request.args.get('price', type = int)
  d['street_address'] = request.args.get('street_address', type = str, default = '')
  d['created'] = request.args.get('created', type = str, default = '')

  my_df = pd.DataFrame({k: [v] for k, v in d.items()})
  my_df = my_df.fillna(0)

  feature_extractor = FeatureExtractor(my_df)
  x, x_featurenames = feature_extractor.get_features_pred_instances(my_df, model_feature_names)
  
  print(d)
  for i, fn in enumerate(x_featurenames):
    print("{:s} --> {:}".format(fn, x.iloc[0,i]))
    
  index_to_class =  {int(k):v for k,v in json.loads(model.attr('index_to_class')).items()}
 
  pred = model.predict(xgb.DMatrix(x.values, feature_names = x_featurenames))[0]
  response = jsonify(InterestPrediction(pred, index_to_class).serialize())
  response.status_code = 200 
  return response

# default route
@app.route('/')
def index():
  return escape('Apartment interest predictor') 

def create_app():
  app.debug = False
  load_model()
  return app

if __name__ == '__main__':
  app = create_app()
  app.run(debug=True, use_reloader=False)
  
'''
cd src
PYTHONPATH=. python service/app.py 
'''
  
  # http://127.0.0.1:5000/interest_prediction?bedrooms=1&bathrooms=5&latitude=40.7&longitude=-73.9425&price=200&feature=a&feature=b&feature=c&feature=d&description=%22a%20b%20c%20d%20e%20f%20g%22&photo=b&photo=b&photo=c&photo=d
  
  
  
  
  
  
  
