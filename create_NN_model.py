import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib
import time

print("\n*** START ***")
# Record the start time
start_time = time.time()

# Carrega o dataset
dataset = pd.read_csv('./datasets/DATASET_4SINGERS_CLEAN.csv')

# y = dataset.iloc[:, 0].to_numpy()   # Variavel Target (singer)
# X = dataset.iloc[:, 1:].to_numpy()  # Features

X = dataset.drop('singer', axis=1).values # Variavel Target (singer)
y = dataset['singer'].values # Features

print('y',y)
print('\nX',X)

# Dividir os dados em treino (80%) e teste (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar o modelo de Rede Neural com os parâmetros especificados
mlp = MLPClassifier(hidden_layer_sizes=(1000, 1000, 1000),
                    activation='relu',
                    solver='adam',
                    alpha=0.0001,
                    max_iter=150000,
                    random_state=42)

# Treinar o modelo
mlp.fit(X_train, y_train)

# Avaliar o modelo
y_pred = mlp.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f'Acurácia no conjunto de teste: {accuracy:.2f}')
print(f'F1 Score no conjunto de teste: {f1:.2f}')

# Salvar o modelo treinado
model_name = 'NN_model'
joblib.dump(mlp, f'{model_name}.pkl')
print(f'Modelo salvo como {model_name}.pkl')

# Record the end time
end_time = time.time()
# Calculate the elapsed time
elapsed_time = end_time - start_time
# Print the elapsed time
print(f"Elapsed time: {elapsed_time} seconds")
print("*** END ***\n")
