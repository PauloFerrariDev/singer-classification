import numpy as np
import tensorflow as tf
import joblib

# Load the saved model
model = tf.keras.models.load_model('best_model.h5')

# Load validation data
X_val, y_val = joblib.load('validation_data.pkl')

# Make predictions
predictions = (model.predict(X_val) > 0.5).astype(int)

# Evaluate predictions (example using F1 score)
from sklearn.metrics import f1_score
f1 = f1_score(y_val, predictions)
print(f"F1 Score: {f1:.4f}")
