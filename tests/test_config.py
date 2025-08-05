import  pytest
from predictions.predict import predict
import numpy as np
input_data = {'original_data':
    {'Open': 40.6,
    'High': 40.76,
    'Low': 40.31,
    'Close': 40.72},

}

target_data = {'min': 101612,
               'max': 135227059}



#def test_difference(data = input_data['original_data']):
#    new_data = np.array([list(data.values())])
#    res = predict(new_data)
#    assert target_data['min'] <= res[0] <= target_data['max']



