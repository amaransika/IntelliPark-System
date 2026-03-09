
import os
import pytest
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(base_dir, 'models', 'best_parking_model.keras')

@pytest.fixture(scope="module")
def parking_ai_setup():
    assert os.path.exists(MODEL_PATH), "Error: CNN Model file not found!"
    
    model = tf.keras.models.load_model(MODEL_PATH)
    
    lb = LabelBinarizer()
    lb.fit(['OVERCAST', 'RAINY', 'SUNNY'])
    
    return model, lb

def test_multi_modal_feature_fusion(parking_ai_setup):
    model, lb = parking_ai_setup
    
    dummy_image = np.random.rand(1, 150, 150, 3).astype('float32')
    
    weather_vector = lb.transform(['RAINY'])[0].astype('float32')
    dummy_weather = np.array([weather_vector]) 
    
    prediction = model.predict([dummy_image, dummy_weather], verbose=0)
    
    assert prediction is not None, "Model failed to return a prediction"
    assert prediction.shape == (1, 1), "Output shape should be a single probability value"
    
    confidence = float(prediction[0][0])
    assert 0.0 <= confidence <= 1.0, f"Confidence value {confidence} is out of bounds!"