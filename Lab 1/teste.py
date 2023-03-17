import matplotlib.pyplot as plt #Importa a biblioteca matplotlib associada ao comando plt com a função de desenhar gráficos
import pickle as p1 #importa a biblioteca pickle associada ao comando p1 com a função de serializar o código.
import numpy as np #importa a biblioteca NumPy associada ao comando np com a função de suporte a grandes Datasets multi-dimensionais e matrizes assim como uma grande coleção de funções matemáticas de alto nível.
import pandas as pd #importa a biblioteca pandas associada ao comando pd com a função de facilitar a manipulação de dados a partir de diversos tipos de entrada.
from sklearn import datasets, linear_model #importa a biblioteca sklearn com as funções 1- Armazenar conjunto de datasets que podem ser importados. 2- de implementar varios modelos de calculos lineares
from sklearn.metrics import mean_squared_error, r2_score #importa a biblioteca como ferramentas de coeficiente de corelação


data = pd.read_csv('dataset/optdigits.tes', sep=',', header=None)
evalution_data=data[:]
data_X=evalution_data.iloc[:,0:64]
print(data_X)
data_Y=evalution_data.iloc[:,64:65]
loaded_model = p1.load(open('dataset/character_predictor', 'rb'))
print('Coeficientes:\n', loaded_model.coef_)
print('=============')
y_pred=loaded_model.predict(data_X)
z_pred=y_pred-data_Y
right = 0
wrong = 0
total = 0
for i in z_pred[64]:
    z=int(i)
    total=total+1
    if z==0:
        right += 1
    else: wrong +=1
print('Frequencia de Certos:', right/total, 'Frequencia de errados:', wrong/total)
