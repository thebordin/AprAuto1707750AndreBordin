import matplotlib.pyplot as plt #Importa a biblioteca matplotlib associada ao comando plt com a função de desenhar gráficos
import pickle as p1 #importa a biblioteca pickle associada ao comando p1 com a função de serializar o código.
import numpy as np #importa a biblioteca NumPy associada ao comando np com a função de suporte a grandes Datasets multi-dimensionais e matrizes assim como uma grande coleção de funções matemáticas de alto nível.
import pandas as pd #importa a biblioteca pandas associada ao comando pd com a função de facilitar a manipulação de dados a partir de diversos tipos de entrada.
from sklearn import datasets, linear_model #importa a biblioteca sklearn com as funções 1- Armazenar conjunto de datasets que podem ser importados. 2- de implementar varios modelos de calculos lineares
from sklearn.metrics import mean_squared_error, r2_score #importa a biblioteca como ferramentas de coeficiente de corelação

dataxxx = pd.read_csv('dataset/optdigits.tra', sep=',', header=None)
data=dataxxx.iloc[5:6,0:64]
print('data', data)
'''fmap_data = map(float,data)
print('fmap=',fmap_data)
flist_data = list(fmap_data)
print('flist=',flist_data)
data1 = pd.read_csv('dataset/optdigits.tra', sep=',', header=None)
print('data1', data1)
data2=data1.iloc[:1,:64]
print ('data2=',data2)
data_prep=pd.DataFrame([data],columns=list(data2))
print ('dataprep=',data_prep)'''
out=data
for i in out:
    print(i, data[i])
    loaded_model = p1.load(open('dataset/character_predictor', 'rb'))
    y_prep=loaded_model.predict(data)
    print('Caractere: ',int(y_prep))