import matplotlib.pyplot as plt #Importa a biblioteca matplotlib associada ao comando plt com a função de desenhar gráficos
import pickle as p1 #importa a biblioteca pickle associada ao comando p1 com a função de serializar o código.
import numpy as np #importa a biblioteca NumPy associada ao comando np com a função de suporte a grandes Datasets multi-dimensionais e matrizes assim como uma grande coleção de funções matemáticas de alto nível.
import pandas as pd #importa a biblioteca pandas associada ao comando pd com a função de facilitar a manipulação de dados a partir de diversos tipos de entrada.
from sklearn import datasets, linear_model #importa a biblioteca sklearn com as funções 1- Armazenar conjunto de datasets que podem ser importados. 2- de implementar varios modelos de calculos lineares
from sklearn.metrics import mean_squared_error, r2_score #importa a biblioteca como ferramentas de coeficiente de corelação
import random
charReal = 1
charPrevisto = 2
contador=0
while charReal != charPrevisto:
    contador += 1
    linha = random.randint(0,3824)
    dataPre = pd.read_csv('dataset/optdigits.tra', sep=',', header=None)
    data=dataPre.iloc[linha:linha+1,0:64]
    #print('data\n', data)
    #print('dataPre\n', dataPre.iloc[linha:linha+1,64])
    charReal=int(dataPre.iloc[linha:linha+1,64])
    out=data
    loaded_model = p1.load(open('dataset/character_predictor', 'rb'))
    charPrevisto = int(loaded_model.predict(data))
    #print(charPrevisto)
    if charReal == charPrevisto:
        print('Previsão CORRETA <==============================================================================')
    else: print('Previsão INCORRETA')
    print('linha: ', linha)
    print('Caractere Real: ',charReal,'Caractere Previsto: ', int(charPrevisto),'\n')
print('Foram ', contador, 'tentativas até um match.')