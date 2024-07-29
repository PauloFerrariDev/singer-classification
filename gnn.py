import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# Load the dataset
data = pd.read_csv('./datasets/DATASET_4SINGERS_CLEAN.csv')

# Separate features and target
X = data.drop('singer', axis=1).values.astype('float32')
y = data['singer'].values

print('y',y)
print('\nX',X)

# Encode the target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Define the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_split=0.2)

# Save the model
model.save('neural_network_model.h5')

# Save the label encoder for future use
joblib.dump(label_encoder, 'label_encoder.pkl')

# Save validation data for later use
joblib.dump((X_test, y_test), 'validation_data.pkl')
