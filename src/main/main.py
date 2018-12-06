from model.extract_features import FeatureExtractor
import os

data_dir = "../../data/rental_listings"
input_dir = os.path.join(data_dir, "input")

train_filename = os.path.join(input_dir,"train.json")

featureExtractor = FeatureExtractor(train_filename, max_rows=100, target_variable="interest_level")

df = featureExtractor.get_features()

print(df)

#print(featureExtractor.get_target())
