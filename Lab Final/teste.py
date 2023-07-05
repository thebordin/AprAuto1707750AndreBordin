import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense

# Carregar conjunto de dados Iris
data = load_iris()
X = data.data
y = data.target

# Pré-processamento dos dados
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dividir os dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Etapa de pré-treinamento - Autoencoder
autoencoder = tf.keras.models.Sequential([
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(64, activation='relu'),
    Dense(X.shape[1])
])
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_train, X_train, epochs=100, batch_size=16)

# Extrair as representações latentes dos dados de treinamento
encoder = tf.keras.models.Sequential(autoencoder.layers[:3])

X_train_encoded = encoder.predict(X_train)
X_test_encoded = encoder.predict(X_test)

# Afinamento supervisionado - MLPC
mlpc = tf.keras.models.Sequential([
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(len(np.unique(y_train)), activation='softmax')
])
mlpc.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
mlpc.fit(X_train_encoded, y_train, epochs=50, batch_size=16)

# Avaliar o desempenho do modelo afinado
_, accuracy = mlpc.evaluate(X_test_encoded, y_test)
print('Acurácia do modelo afinado:', accuracy)