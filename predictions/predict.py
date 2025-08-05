import joblib
import numpy as np
import yaml
import json
config_path = 'params.yaml'
with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
schema_in = config['schema']['schema_in']


class NotInRange(Exception):
     def __init__(self, message = 'The value is not in range'):
          self.message = message
          super().__init__(self.message)

def get_schema(schema = schema_in):
    with open(schema) as m:
        schema = json.load(m)
    return schema

def validate(col, val):
     schema = get_schema()

     try:
         if schema[col]['min'] <= val <= get_schema()[col]['max']:
              return val
         else: 
              pass
                
           # else:
             #   raise NotInRange(f"The value for {col} is not in the range {schema[col]['min']} to {schema[col]['max']}")
     except NotInRange as e:
          print(str(e.message))
              

def predict(features):
    
    model_path = config['model_dir']['model']
    scaler_path = config['model_dir']['scaler']
    with open(model_path, 'rb') as r:
        model = joblib.load(r)
    with open(scaler_path, 'rb') as s:
        scaler = joblib.load(s)

    features= scaler.transform(features)
    prediction = model.predict(features)
    return prediction
