import pandas as pd
import numpy as np
from sklearn import neighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NeighborhoodComponentsAnalysis
import matplotlib.pyplot as plt

# Gerar um número aleatório entre 0 e 3822
rng = np.random.randint(0, 3822)

# Definir o número de vizinhos mais próximos
n_neighbors = 15

# Importar o conjunto de dados
dataset = pd.read_csv('dataset/optdigits.tra', sep=',', header=None)
xyraw = dataset[:]

# Separar as variáveis dependentes e independentes
X = xyraw.iloc[:, 0:64]
y = xyraw.iloc[:, 64]

# Selecionar a observação de teste
X_test = xyraw.iloc[rng, 0:64]
y_test = xyraw.iloc[rng, 64]

# Padronizar as entradas
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform([X_test])

# Reduzir as dimensões das entradas
nca = make_pipeline(StandardScaler(), NeighborhoodComponentsAnalysis(n_components=2, random_state=None))
X_nca = nca.fit_transform(X_scaled, y)

# Treinar o modelo de KNN
knn = neighbors.KNeighborsClassifier(n_neighbors)
knn.fit(X_nca, y)

# Fazer a previsão para a observação de teste
X_test_nca = nca.transform(X_test_scaled)
y_pred = knn.predict(X_test_nca)[0]

# Obter o valor esperado da observação de teste
y_esperado = y_test

# Plotar o mapa de cores
fig, ax = plt.subplots()
scatter = ax.scatter(X_nca[:, 0], X_nca[:, 1], c=y, cmap='Paired')

# Adicionar o marcador para a observação de teste
ax.scatter(X_test_nca[:, 0], X_test_nca[:, 1], marker='*', c='red', s=100)

# Adicionar uma legenda
legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
ax.add_artist(legend1)

# Adicionar o título e exibir o gráfico
if y_pred == y_esperado:
    ax.set_title(f"Previsão correta para a observação {rng}")
else:
    ax.set_title(f"Previsão incorreta para a observação {rng}")
plt.show()