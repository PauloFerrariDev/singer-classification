import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import time

print("\n*** START ***")
# Record the start time
start_time = time.time()

# Load the dataset
dataset = pd.read_csv('./datasets/DATASET_4SINGERS_CLEAN.csv')

y = dataset.iloc[:, 0].to_numpy()   # Target variable (singer)
X = dataset.iloc[:, 1:].to_numpy()  # Features

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Handle imbalanced data using SMOTE
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

# Split the data into train (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the model with specified parameters
mlp = MLPClassifier(hidden_layer_sizes=(1000, 800, 500, 100),
                    activation='relu',
                    solver='adam',
                    alpha=0.0001,
                    max_iter=15000,
                    random_state=42)

# Hyperparameter tuning with Grid Search
param_grid = {
    'hidden_layer_sizes': [(500,), (1000,), (500, 500), (1000, 500)],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate_init': [0.001, 0.01, 0.1]
}

grid_search = GridSearchCV(mlp, param_grid, scoring='f1_weighted', cv=3)
grid_search.fit(X_train, y_train)

# Best model from grid search
best_mlp = grid_search.best_estimator_

# Evaluate the model
y_pred = best_mlp.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f'Best Params: {grid_search.best_params_}')
print(f'Accuracy on test set: {accuracy:.2f}')
print(f'F1 Score on test set: {f1:.2f}')

# Save the trained model
model_name = 'NN_model2'
joblib.dump(best_mlp, f'{model_name}.pkl')
print(f'Model saved as {model_name}.pkl')

# Record the end time
end_time = time.time()
# Calculate the elapsed time
elapsed_time = end_time - start_time
# Print the elapsed time
print(f"Elapsed time: {elapsed_time} seconds")
print("*** END ***\n")
