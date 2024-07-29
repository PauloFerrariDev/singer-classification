import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import joblib

# Carregar o modelo treinado
model = load_model('NN_model_tensorflow.h5')

# Carregar o scaler e o label encoder salvos
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Carregar o novo conjunto de dados
novo_dataset = pd.read_csv('novo_dataset.csv')

# Pré-processamento: padronizar as características do novo conjunto de dados
X_novo = novo_dataset.iloc[:, 3:]  # Ajuste as colunas conforme necessário
X_novo = scaler.transform(X_novo)

# Fazer previsões no novo conjunto de dados
y_novo_pred = model.predict(X_novo)
y_novo_pred_classes = y_novo_pred.argmax(axis=-1)
y_novo_pred_labels = label_encoder.inverse_transform(y_novo_pred_classes)

# Adicionar as previsões ao DataFrame original para análise
novo_dataset['Previsoes'] = y_novo_pred_labels

# Salvar o DataFrame com as previsões
novo_dataset.to_csv('novo_dataset_com_previsoes.csv', index=False)

# Exibir as previsões
print(novo_dataset)
