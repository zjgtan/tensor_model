import tensorflow as tf
import joblib
from sklearn.metrics import roc_auc_score

write_features_map_path = "../mtl/" + 'features_map.pkl'


vocabulary = joblib.load(write_features_map_path)

print(vocabulary.keys())

